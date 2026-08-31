from __future__ import annotations

import atexit
import base64
import copy
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError
from torch.nn import functional as F

from repostguard.config import config_digest
from repostguard.data.transforms import apply_transform, harmonize_image_format, to_model_tensor
from repostguard.models import build_model, count_parameters

from .schemas import RobustnessRequest
from .settings import (
    CALIBRATION_PATH,
    IMAGE_CACHE_SIZE,
    JOB_TTL_SECONDS,
    MAX_FILE_BYTES,
    MAX_IMAGE_PIXELS,
    MODEL_SPECS,
    SUPPORTED_FORMATS,
)


ImageFile.LOAD_TRUNCATED_IMAGES = False


class DemoInputError(ValueError):
    """An image or path is invalid for the local demo contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_relative_path(value: str) -> str:
    normalized = value.replace("\\", "/").strip()
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or normalized in {".", ".."}:
        raise DemoInputError("image_path must be a non-empty relative path")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise DemoInputError(f"Unsafe relative path: {value!r}")
    return path.as_posix()


def decode_image_bytes(payload: bytes) -> tuple[Image.Image, dict[str, Any]]:
    if not payload:
        raise DemoInputError("The uploaded file is empty")
    if len(payload) > MAX_FILE_BYTES:
        raise DemoInputError(f"The uploaded file exceeds {MAX_FILE_BYTES // (1024 * 1024)} MB")
    try:
        with Image.open(io.BytesIO(payload)) as source:
            decoded_format = str(source.format or "").upper()
            if decoded_format not in SUPPORTED_FORMATS:
                raise DemoInputError(f"Unsupported decoded image format: {decoded_format or 'unknown'}")
            stored_width, stored_height = source.size
            if (
                stored_width <= 0
                or stored_height <= 0
                or stored_width * stored_height > MAX_IMAGE_PIXELS
            ):
                raise DemoInputError(
                    f"Decoded image dimensions are not allowed: {stored_width}×{stored_height}"
                )
            first_frame_only = bool(getattr(source, "is_animated", False)) or decoded_format == "MPO"
            if first_frame_only:
                source.seek(0)
            # Camera JPEG/MPO files commonly store portrait pixels sideways and
            # rely on EXIF orientation for display. Apply that orientation before
            # the repository's fixed resize/Q90 model preprocessing so inference,
            # robustness previews, heatmaps, and reported dimensions agree.
            oriented = ImageOps.exif_transpose(source)
            width, height = oriented.size
            image = oriented.convert("RGB").copy()
    except DemoInputError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise DemoInputError("The uploaded file is not a readable supported image") from error
    return image, {
        "width": width,
        "height": height,
        "format": "JPEG" if decoded_format == "MPO" else decoded_format,
        "bytes": len(payload),
        "animated_first_frame": first_frame_only,
    }


def prepare_clean_image(image: Image.Image) -> Image.Image:
    return harmonize_image_format(image, 224, quality=90, jpeg_subsampling=2)


def image_data_url(image: Image.Image, *, image_format: str = "JPEG") -> str:
    buffer = io.BytesIO()
    if image_format == "JPEG":
        image.convert("RGB").save(buffer, format="JPEG", quality=90, subsampling=2)
        mime = "image/jpeg"
    else:
        image.save(buffer, format="PNG", optimize=True)
        mime = "image/png"
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def source_preview_data_url(image: Image.Image, *, max_dimension: int = 1024) -> str:
    """Return a bounded, orientation-correct browser preview of the decoded source."""
    preview = image.copy()
    preview.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
    return image_data_url(preview)


class CalibrationRegistry:
    def __init__(self, path: Path = CALIBRATION_PATH) -> None:
        self.path = path
        self._payload: dict[str, Any] = {}
        self._status: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as error:
            for name in MODEL_SPECS:
                self._status[name] = {"available": False, "reason": f"Calibration unreadable: {error}"}
            return
        self._payload = payload
        recorded = dict(payload.get("teacher_checkpoint_sha256", {}))
        temperatures = dict(payload.get("temperatures", {}))
        common_valid = (
            payload.get("samples") == 2000
            and payload.get("views") == 4
            and bool(payload.get("manifest_sha256"))
            and bool(payload.get("preprocessing_sha256"))
        )
        for name, spec in MODEL_SPECS.items():
            temperature = temperatures.get(name)
            valid_temperature = isinstance(temperature, (float, int)) and math.isfinite(temperature) and temperature > 0
            available = bool(common_valid and valid_temperature and recorded.get(name) == spec.checkpoint_sha256)
            self._status[name] = {
                "available": available,
                "temperature": float(temperature) if valid_temperature else None,
                "reason": None if available else "Calibration lineage does not match the active checkpoint",
                "samples": payload.get("samples"),
                "views": payload.get("views"),
                "manifest_sha256": payload.get("manifest_sha256"),
                "preprocessing_sha256": payload.get("preprocessing_sha256"),
            }

    @staticmethod
    def _temperature_scale(probability: float, temperature: float) -> float:
        clipped = min(max(float(probability), 1e-7), 1.0 - 1e-7)
        logit = math.log(clipped / (1.0 - clipped))
        return 1.0 / (1.0 + math.exp(-logit / temperature))

    def describe(self, name: str) -> dict[str, Any]:
        status = dict(self._status[name])
        if status["available"]:
            status["calibrated_threshold"] = self._temperature_scale(
                MODEL_SPECS[name].threshold, status["temperature"]
            )
        else:
            status["calibrated_threshold"] = None
        return status

    def transform(self, name: str, probability: float) -> tuple[float | None, float | None, float | None]:
        status = self._status[name]
        if not status["available"]:
            return None, None, None
        temperature = float(status["temperature"])
        calibrated = self._temperature_scale(probability, temperature)
        calibrated_threshold = self._temperature_scale(MODEL_SPECS[name].threshold, temperature)
        if calibrated <= 0.0 or calibrated >= 1.0:
            entropy = 0.0
        else:
            entropy = -(
                calibrated * math.log(calibrated)
                + (1.0 - calibrated) * math.log(1.0 - calibrated)
            ) / math.log(2.0)
        return calibrated, calibrated_threshold, entropy


@dataclass
class LoadedModel:
    model: torch.nn.Module
    config: dict[str, Any]
    checkpoint_sha256: str
    parameters: dict[str, int]


class ModelRegistry:
    def __init__(self, calibration: CalibrationRegistry) -> None:
        self.calibration = calibration
        self.device = self._select_device()
        self._models: dict[str, LoadedModel] = {}
        self._verified_sha: dict[str, str] = {}
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._errors: dict[str, str] = {}

    @staticmethod
    def _select_device() -> torch.device:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _verify_file(self, name: str) -> str:
        spec = MODEL_SPECS[name]
        if not spec.checkpoint.is_file():
            raise FileNotFoundError(f"Missing checkpoint for {name.upper()}")
        digest = self._verified_sha.get(name)
        if digest is None:
            digest = sha256_file(spec.checkpoint)
            self._verified_sha[name] = digest
        if digest != spec.checkpoint_sha256:
            raise ValueError(f"Checkpoint SHA-256 mismatch for {name.upper()}")
        return digest

    def load(self, name: str) -> LoadedModel:
        if name not in MODEL_SPECS:
            raise DemoInputError(f"Unsupported model: {name}")
        if name in self._models:
            return self._models[name]
        with self._load_lock:
            if name in self._models:
                return self._models[name]
            spec = MODEL_SPECS[name]
            try:
                digest = self._verify_file(name)
                checkpoint = torch.load(spec.checkpoint, map_location="cpu", weights_only=False)
                checkpoint_config = checkpoint.get("config")
                if not isinstance(checkpoint_config, dict):
                    raise TypeError("Checkpoint does not contain an embedded configuration")
                embedded_digest = checkpoint.get("config_sha256")
                if embedded_digest != spec.config_sha256 or config_digest(checkpoint_config) != spec.config_sha256:
                    raise ValueError(f"Embedded config digest mismatch for {name.upper()}")
                if str(checkpoint_config.get("model", {}).get("experiment", "")).lower() != name:
                    raise ValueError(f"Embedded model experiment does not match {name.upper()}")
                train_manifest = str(checkpoint_config.get("data", {}).get("train_manifest", ""))
                if "community_forensics_train_v3.csv" not in train_manifest:
                    raise ValueError(f"{name.upper()} is not a train-v3 checkpoint")
                validation_threshold = float(checkpoint.get("validation_metrics", {}).get("threshold"))
                if not math.isclose(validation_threshold, spec.threshold, abs_tol=1e-12):
                    raise ValueError(f"Frozen threshold mismatch for {name.upper()}")

                runtime_config = copy.deepcopy(checkpoint_config)
                runtime_config["model"]["clip_pretrained"] = None
                model = build_model(runtime_config)
                model.load_state_dict(checkpoint["model"], strict=True)
                model.eval().to(self.device)
                loaded = LoadedModel(
                    model=model,
                    config=checkpoint_config,
                    checkpoint_sha256=digest,
                    parameters=count_parameters(model),
                )
                self._models[name] = loaded
                self._errors.pop(name, None)
                del checkpoint
                return loaded
            except Exception as error:
                self._errors[name] = f"{type(error).__name__}: {error}"
                raise

    def health(self) -> dict[str, Any]:
        models: dict[str, Any] = {}
        for name, spec in MODEL_SPECS.items():
            actual_sha: str | None = self._verified_sha.get(name)
            verification = "verified" if actual_sha == spec.checkpoint_sha256 else "pending"
            if not spec.checkpoint.is_file():
                verification = "missing"
            if name in self._errors:
                verification = "error"
            models[name] = {
                "available": spec.checkpoint.is_file() and name not in self._errors,
                "loaded": name in self._models,
                "verification": verification,
                "error": self._errors.get(name),
                "checkpoint_sha256": spec.checkpoint_sha256,
                "config_sha256": spec.config_sha256,
                "threshold": spec.threshold,
                "clean_auroc": spec.clean_auroc,
                "train_version": "v3",
                "calibration": self.calibration.describe(name),
            }
        return {
            "status": "ok" if any(item["available"] for item in models.values()) else "degraded",
            "device": str(self.device),
            "mps_built": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_built()),
            "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
            "models": models,
        }

    @staticmethod
    def _normalize_map(values: np.ndarray) -> np.ndarray:
        low, high = np.quantile(values, [0.01, 0.99])
        if not np.isfinite(low) or not np.isfinite(high) or high - low < 1e-8:
            return np.zeros_like(values, dtype=np.float32)
        return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _colorize(values: np.ndarray) -> np.ndarray:
        stops = np.asarray(
            [
                [7, 12, 18],
                [22, 90, 120],
                [50, 224, 218],
                [241, 218, 98],
                [255, 93, 78],
            ],
            dtype=np.float32,
        )
        scaled = values * (len(stops) - 1)
        left = np.floor(scaled).astype(np.int32)
        right = np.clip(left + 1, 0, len(stops) - 1)
        fraction = (scaled - left)[..., None]
        return (stops[left] * (1.0 - fraction) + stops[right] * fraction).astype(np.uint8)

    def _heatmaps(self, images: torch.Tensor, model: torch.nn.Module) -> dict[str, str]:
        gray = images.mean(dim=1, keepdim=True)
        filters = model.forensic.srm_filters.to(images.device)
        srm = F.conv2d(gray, filters, padding=2).abs().mean(dim=1)[0]
        reduced = F.interpolate(images, scale_factor=0.5, mode="nearest")
        restored = F.interpolate(reduced, size=images.shape[-2:], mode="nearest")
        npr = (images - restored).abs().mean(dim=1)[0]
        output: dict[str, str] = {}
        for name, tensor in (("srm", srm), ("npr", npr)):
            normalized = self._normalize_map(tensor.detach().float().cpu().numpy())
            gray_pixels = (normalized * 255.0).astype(np.uint8)
            color_pixels = self._colorize(normalized)
            output[f"{name}_gray"] = image_data_url(Image.fromarray(gray_pixels, mode="L"), image_format="PNG")
            output[f"{name}_color"] = image_data_url(Image.fromarray(color_pixels, mode="RGB"), image_format="PNG")
        return output

    @staticmethod
    def _m2_contribution(model: torch.nn.Module, output: dict[str, torch.Tensor]) -> dict[str, Any]:
        semantic = output["semantic_features"]
        forensic = output["forensic_features"]
        full_logit = output["logits"]
        without_semantic = model.classifier(
            model.fusion(torch.cat((torch.zeros_like(semantic), forensic), dim=1))
        ).squeeze(1)
        without_forensic = model.classifier(
            model.fusion(torch.cat((semantic, torch.zeros_like(forensic)), dim=1))
        ).squeeze(1)
        semantic_effect = float((full_logit - without_semantic)[0].detach().cpu())
        forensic_effect = float((full_logit - without_forensic)[0].detach().cpu())
        total = abs(semantic_effect) + abs(forensic_effect)
        low_signal = total < 1e-8
        semantic_share = 0.5 if low_signal else abs(semantic_effect) / total
        return {
            "kind": "ablation",
            "title": "分支消融贡献",
            "semantic": semantic_share,
            "forensic": 1.0 - semantic_share,
            "semantic_logit_effect": semantic_effect,
            "forensic_logit_effect": forensic_effect,
            "low_signal": low_signal,
            "note": "M2 无 quality gate；比例来自移除分支的 logit 变化，不是因果权重。",
        }

    def predict(self, image: Image.Image, name: str) -> dict[str, Any]:
        loaded = self.load(name)
        tensor = to_model_tensor(image, 224).unsqueeze(0).to(self.device)
        started = time.perf_counter()
        with self._inference_lock, torch.inference_mode():
            output = loaded.model(tensor)
            probability = float(torch.sigmoid(output["logits"])[0].float().cpu())
            if name == "m3":
                fractions = output["gate_fractions"][0].float().cpu()
                semantic = float(fractions[0])
                forensic = float(fractions[1])
                branch_evidence = {
                    "kind": "gate",
                    "title": "Quality-aware gate 权重",
                    "semantic": semantic,
                    "forensic": forensic,
                    "low_signal": False,
                    "note": "M3 gate 仅调节语义与取证分支，不将质量特征直接送入分类器。",
                }
            else:
                branch_evidence = self._m2_contribution(loaded.model, output)
            heatmaps = self._heatmaps(tensor, loaded.model)
        calibrated, calibrated_threshold, uncertainty = self.calibration.transform(name, probability)
        label = "AIGC" if probability >= MODEL_SPECS[name].threshold else "Real"
        if calibrated is not None and calibrated_threshold is not None:
            calibrated_label = "AIGC" if calibrated >= calibrated_threshold else "Real"
            if calibrated_label != label:
                raise RuntimeError("Calibration changed a frozen model decision")
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {
            "model": name,
            "checkpoint_sha256": loaded.checkpoint_sha256,
            "raw_score": probability,
            "calibrated_score": calibrated,
            "raw_threshold": MODEL_SPECS[name].threshold,
            "calibrated_threshold": calibrated_threshold,
            "label": label,
            "uncertainty_entropy": uncertainty,
            "branch_evidence": branch_evidence,
            "heatmaps": heatmaps,
            "preview": image_data_url(image),
            "timing_ms": elapsed_ms,
            "device": str(self.device),
            "parameters": loaded.parameters,
        }


class ImageCache:
    def __init__(self, capacity: int = IMAGE_CACHE_SIZE) -> None:
        self.capacity = capacity
        self._items: OrderedDict[str, tuple[Image.Image, dict[str, Any]]] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, image: Image.Image, metadata: dict[str, Any]) -> str:
        image_id = uuid.uuid4().hex
        with self._lock:
            self._items[image_id] = (image.copy(), dict(metadata))
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)
        return image_id

    def get(self, image_id: str) -> tuple[Image.Image, dict[str, Any]]:
        with self._lock:
            item = self._items.get(image_id)
            if item is None:
                raise KeyError("Image session expired or does not exist")
            self._items.move_to_end(image_id)
            return item[0].copy(), dict(item[1])


def apply_robustness(clean: Image.Image, request: RobustnessRequest) -> tuple[Image.Image, list[dict[str, Any]]]:
    image = clean.copy()
    applied: list[dict[str, Any]] = []
    if request.crop.enabled:
        params = {"ratio": request.crop.ratio}
        image = apply_transform(image, "center_crop", params)
        applied.append({"name": "center_crop", "params": params})
    if request.resize.enabled:
        params = {"scale": request.resize.scale, "interpolation": "bicubic"}
        image = apply_transform(image, "resize", params)
        applied.append({"name": "resize", "params": params})
    if request.jitter.enabled:
        params = {
            "brightness": request.jitter.brightness,
            "contrast": request.jitter.contrast,
            "saturation": request.jitter.saturation,
        }
        image = apply_transform(image, "color_jitter", params)
        applied.append({"name": "color_jitter", "params": params})
    if request.blur.enabled:
        params = {"sigma": request.blur.sigma}
        image = apply_transform(image, "gaussian_blur", params)
        applied.append({"name": "gaussian_blur", "params": params})
    if request.noise.enabled:
        params = {"sigma": request.noise.sigma, "seed": 20260827}
        image = apply_transform(image, "gaussian_noise", params)
        applied.append({"name": "gaussian_noise", "params": params})
    if request.jpeg.enabled:
        params = {"quality": request.jpeg.quality, "subsampling": 2}
        image = apply_transform(image, "jpeg", params)
        applied.append({"name": "jpeg", "params": params})
    return image, applied


@dataclass
class BatchJob:
    id: str
    model: str
    total: int
    temp_dir: Path
    created_at: float = field(default_factory=time.time)
    status: str = "queued"
    processed: int = 0
    results: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)

    def public(self) -> dict[str, Any]:
        return {
            "job_id": self.id,
            "model": self.model,
            "status": self.status,
            "total": self.total,
            "processed": self.processed,
            "succeeded": len(self.results),
            "failed": len(self.errors),
            "results": list(self.results),
            "errors": list(self.errors),
        }


class BatchManager:
    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry
        self._jobs: dict[str, BatchJob] = {}
        self._lock = threading.Lock()
        atexit.register(self.close)

    def create(self, model: str, logical_paths: list[str], stored_paths: list[Path], temp_dir: Path) -> BatchJob:
        normalized = [validate_relative_path(path) for path in logical_paths]
        if len(set(normalized)) != len(normalized):
            raise DemoInputError("Duplicate image_path values are not allowed")
        if len(normalized) != len(stored_paths):
            raise DemoInputError("Batch file and path counts differ")
        job = BatchJob(id=uuid.uuid4().hex, model=model, total=len(normalized), temp_dir=temp_dir)
        with self._lock:
            self._prune_locked()
            self._jobs[job.id] = job
        worker = threading.Thread(
            target=self._process,
            args=(job, list(zip(normalized, stored_paths, strict=True))),
            name=f"aigi-batch-{job.id[:8]}",
            daemon=True,
        )
        worker.start()
        return job

    def _process(self, job: BatchJob, items: list[tuple[str, Path]]) -> None:
        job.status = "running"
        try:
            for logical_path, stored_path in items:
                try:
                    payload = stored_path.read_bytes()
                    image, _ = decode_image_bytes(payload)
                    clean = prepare_clean_image(image)
                    result = self.registry.predict(clean, job.model)
                    job.results.append({"image_path": logical_path, "pred": round(result["raw_score"], 4)})
                except Exception as error:
                    job.errors.append(
                        {
                            "image_path": logical_path,
                            "status": "error",
                            "error_type": type(error).__name__,
                            "message": str(error),
                        }
                    )
                finally:
                    job.processed += 1
            job.results.sort(key=lambda row: row["image_path"])
            job.status = "complete"
        except Exception as error:
            job.status = "failed"
            job.errors.append(
                {"image_path": "", "status": "error", "error_type": type(error).__name__, "message": str(error)}
            )
        finally:
            shutil.rmtree(job.temp_dir, ignore_errors=True)

    def get(self, job_id: str) -> BatchJob:
        with self._lock:
            self._prune_locked()
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError("Batch job expired or does not exist")
            return job

    def _prune_locked(self) -> None:
        now = time.time()
        expired = [
            job_id
            for job_id, job in self._jobs.items()
            if job.status in {"complete", "failed"} and now - job.created_at > JOB_TTL_SECONDS
        ]
        for job_id in expired:
            job = self._jobs.pop(job_id)
            shutil.rmtree(job.temp_dir, ignore_errors=True)

    @staticmethod
    def new_temp_dir() -> Path:
        return Path(tempfile.mkdtemp(prefix="aigi-detect-demo-"))

    def close(self) -> None:
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            shutil.rmtree(job.temp_dir, ignore_errors=True)


calibration_registry = CalibrationRegistry()
model_registry = ModelRegistry(calibration_registry)
image_cache = ImageCache()
batch_manager = BatchManager(model_registry)
