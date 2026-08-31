from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.nn import functional as F

from repostguard.losses import symmetric_bernoulli_kl


def validate_distillation_config(config: dict[str, Any]) -> None:
    raw = config["distillation"]
    m2_weight = float(raw["m2_weight"])
    m3_weight = float(raw["m3_weight"])
    if not 0.0 <= m2_weight <= 1.0 or not 0.0 <= m3_weight <= 1.0:
        raise ValueError("distillation teacher weights must be in [0, 1]")
    if not math.isclose(m2_weight + m3_weight, 1.0, abs_tol=1e-8):
        raise ValueError("distillation m2_weight and m3_weight must sum to 1")
    feature_weight = float(raw.get("feature_weight", 0.0))
    loss_weights = (
        float(raw["hard_label_weight"]),
        float(raw["soft_teacher_weight"]),
        float(raw["consistency_weight"]),
        feature_weight,
    )
    if any(value < 0.0 for value in loss_weights):
        raise ValueError("distillation loss weights must be non-negative")
    if not math.isclose(sum(loss_weights), 1.0, abs_tol=1e-8):
        raise ValueError("distillation loss weights must sum to 1")
    feature_config = dict(raw.get("feature_distillation", {}))
    feature_enabled = bool(feature_config.get("enabled", False))
    if feature_enabled != (feature_weight > 0.0):
        raise ValueError(
            "feature_distillation.enabled must agree with a positive feature_weight"
        )
    if feature_enabled:
        teacher_dim = int(feature_config.get("teacher_dim", 0))
        if teacher_dim <= 0:
            raise ValueError("feature_distillation.teacher_dim must be positive")
        branch_weights = dict(feature_config.get("branch_weights", {}))
        expected_branches = {"semantic", "forensic", "fused"}
        if set(branch_weights) != expected_branches:
            raise ValueError(
                "feature_distillation.branch_weights must define semantic, forensic, fused"
            )
        values = [float(branch_weights[name]) for name in sorted(expected_branches)]
        if any(value < 0.0 for value in values) or not math.isclose(
            sum(values), 1.0, abs_tol=1e-8
        ):
            raise ValueError("feature branch weights must be non-negative and sum to 1")
    if float(raw["temperature"]) <= 0.0:
        raise ValueError("distillation temperature must be positive")
    calibration_config = dict(raw.get("teacher_calibration", {}))
    calibration_method = str(calibration_config.get("method", "temperature"))
    if calibration_method not in {"temperature", "per_view_affine_platt"}:
        raise ValueError(
            "distillation.teacher_calibration.method must be temperature or "
            "per_view_affine_platt"
        )
    if calibration_method == "per_view_affine_platt":
        calibration_path = str(calibration_config.get("path", "")).strip()
        if not calibration_path:
            raise ValueError(
                "per_view_affine_platt calibration requires "
                "distillation.teacher_calibration.path"
            )
        if not math.isclose(float(raw["temperature"]), 1.0, abs_tol=1e-8):
            raise ValueError(
                "per-view affine teacher targets require distillation.temperature=1 "
                "to avoid applying a second softening"
            )
    reliability_config = dict(raw.get("teacher_reliability", {}))
    if bool(reliability_config.get("enabled", False)):
        if calibration_method != "per_view_affine_platt":
            raise ValueError(
                "teacher reliability weighting requires per-view affine calibration"
            )
        wrong_scale = float(reliability_config.get("wrong_teacher_scale", 0.1))
        confidence_power = float(reliability_config.get("confidence_power", 1.0))
        minimum_weight = float(reliability_config.get("minimum_weight", 0.0))
        if not 0.0 <= wrong_scale <= 1.0:
            raise ValueError("teacher_reliability.wrong_teacher_scale must be in [0, 1]")
        if not math.isfinite(confidence_power) or confidence_power <= 0.0:
            raise ValueError("teacher_reliability.confidence_power must be positive")
        if not 0.0 <= minimum_weight <= 1.0:
            raise ValueError("teacher_reliability.minimum_weight must be in [0, 1]")
    warmup_epochs = int(raw.get("distillation_warmup_epochs", 0))
    if warmup_epochs < 0:
        raise ValueError("distillation_warmup_epochs must be non-negative")
    for teacher in ("m2", "m3"):
        if float(raw["teacher_calibration_temperatures"][teacher]) <= 0.0:
            raise ValueError(f"{teacher} calibration temperature must be positive")
    expected_lineage = raw.get("teacher_checkpoint_sha256", {})
    for teacher in ("m2", "m3"):
        digest = str(expected_lineage.get(teacher, ""))
        if digest and (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest.lower())
        ):
            raise ValueError(f"{teacher} teacher checkpoint SHA256 is invalid")
    threshold = float(raw["disagreement_threshold"])
    scale = float(raw["disagreement_scale"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("disagreement_threshold must be in [0, 1]")
    if not 0.0 <= scale <= 1.0:
        raise ValueError("disagreement_scale must be in [0, 1]")

    if feature_enabled:
        component_weights = dict(
            feature_config.get("component_weights", {"pointwise": 1.0})
        )
        allowed_components = {"pointwise", "relational", "quality_gate"}
        if not component_weights or not set(component_weights).issubset(
            allowed_components
        ):
            raise ValueError(
                "feature_distillation.component_weights may define pointwise, "
                "relational, and quality_gate"
            )
        component_values = [float(value) for value in component_weights.values()]
        if any(value < 0.0 for value in component_values) or not math.isclose(
            sum(component_values), 1.0, abs_tol=1e-8
        ):
            raise ValueError(
                "feature distillation component weights must be non-negative and sum to 1"
            )
        relational_config = dict(feature_config.get("relational", {}))
        relational_enabled = bool(relational_config.get("enabled", False))
        if relational_enabled != (float(component_weights.get("relational", 0.0)) > 0.0):
            raise ValueError(
                "feature_distillation.relational.enabled must agree with its "
                "positive component weight"
            )
        if relational_enabled:
            relational_branch_weights = dict(
                relational_config.get(
                    "branch_weights", {"forensic": 0.4, "fused": 0.6}
                )
            )
            if not relational_branch_weights or not set(
                relational_branch_weights
            ).issubset({"semantic", "forensic", "fused"}):
                raise ValueError("relational branch weights contain an unknown branch")
            relational_values = [
                float(value) for value in relational_branch_weights.values()
            ]
            if any(value < 0.0 for value in relational_values) or not math.isclose(
                sum(relational_values), 1.0, abs_tol=1e-8
            ):
                raise ValueError(
                    "relational branch weights must be non-negative and sum to 1"
                )
            cross_view_weight = float(
                relational_config.get("cross_view_delta_weight", 0.0)
            )
            if not 0.0 <= cross_view_weight <= 1.0:
                raise ValueError(
                    "relational.cross_view_delta_weight must be in [0, 1]"
                )
        gate_enabled = bool(
            feature_config.get("quality_gate_distillation", {}).get(
                "enabled", False
            )
        )
        if gate_enabled != (float(component_weights.get("quality_gate", 0.0)) > 0.0):
            raise ValueError(
                "quality_gate_distillation.enabled must agree with its positive "
                "component weight"
            )


def canonical_view_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    validate_distillation_config(config)
    views = [dict(view) for view in config["distillation"]["views"]]
    if len(views) < 2:
        raise ValueError("At least clean and one augmented distillation view are required")
    identifiers = [str(view["id"]) for view in views]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Distillation view ids must be unique")
    if str(views[0]["id"]) != "clean" or str(views[0]["name"]) != "clean":
        raise ValueError("The first distillation view must be id=clean, name=clean")
    return json.loads(json.dumps(views, sort_keys=True))


@lru_cache(maxsize=16)
def load_affine_teacher_calibration(path: str) -> dict[str, Any]:
    calibration_path = Path(path).expanduser().resolve()
    with calibration_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if str(payload.get("calibration_method", "")) != "per_view_affine_platt":
        raise ValueError(
            f"Calibration file is not per_view_affine_platt: {calibration_path}"
        )
    parameters = payload.get("affine_calibration")
    if not isinstance(parameters, dict):
        raise ValueError(f"Calibration file has no affine_calibration: {calibration_path}")
    return payload


def _view_affine_parameters(
    calibration_payload: dict[str, Any],
    *,
    teacher: str,
    view_ids: Sequence[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    teachers = calibration_payload["affine_calibration"]
    if teacher not in teachers:
        raise KeyError(f"Affine calibration is missing teacher {teacher}")
    teacher_parameters = teachers[teacher]
    slopes: list[float] = []
    intercepts: list[float] = []
    for view_id in view_ids:
        identifier = str(view_id)
        if identifier not in teacher_parameters:
            raise KeyError(
                f"Affine calibration for {teacher} is missing view {identifier!r}"
            )
        row = teacher_parameters[identifier]
        slope = float(row["a"])
        intercept = float(row["b"])
        if not math.isfinite(slope) or slope <= 0.0 or not math.isfinite(intercept):
            raise ValueError(
                f"Invalid affine calibration for {teacher}/{identifier}: "
                f"a={slope}, b={intercept}"
            )
        slopes.append(slope)
        intercepts.append(intercept)
    return (
        torch.tensor(slopes, dtype=torch.float32, device=device),
        torch.tensor(intercepts, dtype=torch.float32, device=device),
    )


def affine_calibrated_teacher_probabilities(
    logits: torch.Tensor,
    *,
    teacher: str,
    view_ids: Sequence[str],
    calibration_payload: dict[str, Any],
) -> torch.Tensor:
    flattened = logits.float().reshape(-1)
    if len(view_ids) != flattened.numel():
        raise ValueError(
            "Teacher logits and per-sample view ids must align: "
            f"{flattened.numel()} != {len(view_ids)}"
        )
    slopes, intercepts = _view_affine_parameters(
        calibration_payload,
        teacher=teacher,
        view_ids=view_ids,
        device=flattened.device,
    )
    return torch.sigmoid(slopes * flattened + intercepts).clamp(
        1e-6, 1.0 - 1e-6
    )


def _validate_calibration_lineage(
    calibration_payload: dict[str, Any], config: dict[str, Any]
) -> None:
    expected = config["distillation"].get("teacher_checkpoint_sha256", {})
    observed = calibration_payload.get("teacher_checkpoint_sha256", {})
    for teacher in ("m2", "m3"):
        expected_digest = str(expected.get(teacher, ""))
        observed_digest = str(observed.get(teacher, ""))
        if expected_digest and observed_digest != expected_digest:
            raise ValueError(
                f"Affine calibration {teacher.upper()} checkpoint lineage mismatch"
            )
    expected_view_ids = [str(view["id"]) for view in canonical_view_specs(config)]
    if calibration_payload.get("view_ids") != expected_view_ids:
        raise ValueError("Affine calibration view lineage mismatch")


def calibrated_teacher_probabilities(
    m2_logits: torch.Tensor,
    m3_logits: torch.Tensor,
    *,
    m2_weight: float,
    m3_weight: float,
    m2_calibration_temperature: float,
    m3_calibration_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    m2_probability = torch.sigmoid(m2_logits.float() / float(m2_calibration_temperature))
    m3_probability = torch.sigmoid(m3_logits.float() / float(m3_calibration_temperature))
    mixed = float(m2_weight) * m2_probability + float(m3_weight) * m3_probability
    disagreement = (m3_probability - m2_probability).abs()
    return mixed.clamp(1e-6, 1.0 - 1e-6), disagreement


def probability_to_logit(probability: torch.Tensor) -> torch.Tensor:
    probability = probability.float().clamp(1e-6, 1.0 - 1e-6)
    return torch.log(probability) - torch.log1p(-probability)


def binary_logit_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    sample_weights: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    temperature_value = float(temperature)
    teacher_targets = torch.sigmoid(teacher_logits.float() / temperature_value)
    losses = F.binary_cross_entropy_with_logits(
        student_logits.float() / temperature_value,
        teacher_targets,
        reduction="none",
    ) * (temperature_value**2)
    weights = sample_weights.float()
    if losses.shape != weights.shape:
        raise ValueError("Teacher distillation losses and sample weights must align")
    return (losses * weights).mean()


def binary_probability_distillation_loss(
    student_logits: torch.Tensor,
    teacher_probabilities: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    """Bernoulli KL with the same gradient as soft-target BCE at T=1.

    Subtracting the teacher entropy removes the large constant that made the
    old soft-loss logs difficult to interpret.  It does not change gradients.
    """

    targets = teacher_probabilities.float().clamp(1e-6, 1.0 - 1e-6)
    student = student_logits.float().reshape_as(targets)
    weights = sample_weights.float().reshape_as(targets)
    cross_entropy = F.binary_cross_entropy_with_logits(
        student, targets, reduction="none"
    )
    teacher_entropy = -(
        targets * torch.log(targets)
        + (1.0 - targets) * torch.log1p(-targets)
    )
    return ((cross_entropy - teacher_entropy) * weights).mean()


def teacher_reliability_weights(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    *,
    confidence_power: float,
    wrong_teacher_scale: float,
    minimum_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = probabilities.float().clamp(1e-6, 1.0 - 1e-6)
    labels = labels.float().reshape_as(probabilities)
    confidence = (2.0 * (probabilities - 0.5).abs()).clamp(0.0, 1.0)
    confidence = confidence.pow(float(confidence_power))
    teacher_correct = (probabilities >= 0.5) == (labels >= 0.5)
    correctness_scale = torch.where(
        teacher_correct,
        torch.ones_like(confidence),
        torch.full_like(confidence, float(wrong_teacher_scale)),
    )
    weights = (confidence * correctness_scale).clamp_min(float(minimum_weight))
    return weights, teacher_correct


def normalized_feature_distillation_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
) -> torch.Tensor:
    if student_features.shape != teacher_features.shape:
        raise ValueError(
            "Student and teacher feature shapes must match: "
            f"{tuple(student_features.shape)} != {tuple(teacher_features.shape)}"
        )
    student = F.normalize(student_features.float(), dim=1)
    teacher = F.normalize(teacher_features.float(), dim=1)
    return (1.0 - (student * teacher).sum(dim=1)).mean()


def relational_feature_distillation_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
) -> torch.Tensor:
    """Match the batch geometry of teacher and Student representations."""

    if student_features.shape != teacher_features.shape:
        raise ValueError(
            "Student and teacher relational feature shapes must match: "
            f"{tuple(student_features.shape)} != {tuple(teacher_features.shape)}"
        )
    student = F.normalize(student_features.float(), dim=1)
    teacher = F.normalize(teacher_features.float(), dim=1)
    student_relation = student @ student.transpose(0, 1)
    teacher_relation = teacher @ teacher.transpose(0, 1)
    return F.smooth_l1_loss(student_relation, teacher_relation)


def cross_view_feature_delta_loss(
    student_clean: torch.Tensor,
    student_augmented: torch.Tensor,
    teacher_clean: torch.Tensor,
    teacher_augmented: torch.Tensor,
) -> torch.Tensor:
    if not (
        student_clean.shape
        == student_augmented.shape
        == teacher_clean.shape
        == teacher_augmented.shape
    ):
        raise ValueError("Cross-view Student and teacher feature shapes must match")
    student_delta = student_augmented.float() - student_clean.float()
    teacher_delta = teacher_augmented.float() - teacher_clean.float()
    return normalized_feature_distillation_loss(student_delta, teacher_delta)


def compute_dual_teacher_loss(
    config: dict[str, Any],
    labels: torch.Tensor,
    clean_output: dict[str, torch.Tensor],
    augmented_output: dict[str, torch.Tensor],
    batch: dict[str, Any],
    *,
    epoch: int | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    validate_distillation_config(config)
    raw = config["distillation"]
    labels_fp32 = labels.float()
    clean_hard = F.binary_cross_entropy_with_logits(clean_output["logits"], labels_fp32)
    augmented_hard = F.binary_cross_entropy_with_logits(
        augmented_output["logits"], labels_fp32
    )
    hard_loss = 0.5 * (clean_hard + augmented_hard)

    m2_weight = float(raw["m2_weight"])
    m3_weight = float(raw["m3_weight"])
    calibration_config = dict(raw.get("teacher_calibration", {}))
    calibration_method = str(calibration_config.get("method", "temperature"))
    if calibration_method == "per_view_affine_platt":
        calibration_path = str(Path(calibration_config["path"]).expanduser().resolve())
        calibration_payload = load_affine_teacher_calibration(calibration_path)
        _validate_calibration_lineage(calibration_payload, config)
        augmented_view_ids = [str(value) for value in batch.get("view_id", [])]
        batch_size = int(labels_fp32.numel())
        if len(augmented_view_ids) != batch_size:
            raise ValueError(
                "Per-view affine KD requires one batch view_id per sample"
            )
        clean_view_ids = ["clean"] * batch_size

        def calibrated_pair(
            m2_key: str,
            m3_key: str,
            view_ids: Sequence[str],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            active_probabilities: list[tuple[float, torch.Tensor]] = []
            if m2_weight > 0.0:
                active_probabilities.append(
                    (
                        m2_weight,
                        affine_calibrated_teacher_probabilities(
                            batch[m2_key],
                            teacher="m2",
                            view_ids=view_ids,
                            calibration_payload=calibration_payload,
                        ),
                    )
                )
            if m3_weight > 0.0:
                active_probabilities.append(
                    (
                        m3_weight,
                        affine_calibrated_teacher_probabilities(
                            batch[m3_key],
                            teacher="m3",
                            view_ids=view_ids,
                            calibration_payload=calibration_payload,
                        ),
                    )
                )
            if not active_probabilities:
                raise ValueError("At least one affine-calibrated teacher must be active")
            mixed = sum(weight * probability for weight, probability in active_probabilities)
            if len(active_probabilities) == 2:
                disagreement = (
                    active_probabilities[0][1] - active_probabilities[1][1]
                ).abs()
            else:
                disagreement = torch.zeros_like(mixed)
            return mixed.clamp(1e-6, 1.0 - 1e-6), disagreement

        clean_probability, clean_disagreement = calibrated_pair(
            "teacher_m2_logit_clean",
            "teacher_m3_logit_clean",
            clean_view_ids,
        )
        augmented_probability, augmented_disagreement = calibrated_pair(
            "teacher_m2_logit_aug",
            "teacher_m3_logit_aug",
            augmented_view_ids,
        )
    else:
        calibration = raw["teacher_calibration_temperatures"]
        clean_probability, clean_disagreement = calibrated_teacher_probabilities(
            batch["teacher_m2_logit_clean"],
            batch["teacher_m3_logit_clean"],
            m2_weight=m2_weight,
            m3_weight=m3_weight,
            m2_calibration_temperature=float(calibration["m2"]),
            m3_calibration_temperature=float(calibration["m3"]),
        )
        augmented_probability, augmented_disagreement = calibrated_teacher_probabilities(
            batch["teacher_m2_logit_aug"],
            batch["teacher_m3_logit_aug"],
            m2_weight=m2_weight,
            m3_weight=m3_weight,
            m2_calibration_temperature=float(calibration["m2"]),
            m3_calibration_temperature=float(calibration["m3"]),
        )

    threshold = float(raw["disagreement_threshold"])
    reliability_config = dict(raw.get("teacher_reliability", {}))
    reliability_enabled = bool(reliability_config.get("enabled", False))
    if reliability_enabled:
        reliability_kwargs = {
            "confidence_power": float(
                reliability_config.get("confidence_power", 1.0)
            ),
            "wrong_teacher_scale": float(
                reliability_config.get("wrong_teacher_scale", 0.1)
            ),
            "minimum_weight": float(
                reliability_config.get("minimum_weight", 0.0)
            ),
        }
        clean_weights, clean_teacher_correct = teacher_reliability_weights(
            clean_probability, labels_fp32, **reliability_kwargs
        )
        augmented_weights, augmented_teacher_correct = teacher_reliability_weights(
            augmented_probability, labels_fp32, **reliability_kwargs
        )
        clean_downweighted = ~clean_teacher_correct
        augmented_downweighted = ~augmented_teacher_correct
    else:
        downweight = float(raw["disagreement_scale"])
        clean_weights = torch.where(
            clean_disagreement > threshold,
            torch.full_like(clean_disagreement, downweight),
            torch.ones_like(clean_disagreement),
        )
        augmented_weights = torch.where(
            augmented_disagreement > threshold,
            torch.full_like(augmented_disagreement, downweight),
            torch.ones_like(augmented_disagreement),
        )
        clean_teacher_correct = (clean_probability >= 0.5) == (labels_fp32 >= 0.5)
        augmented_teacher_correct = (augmented_probability >= 0.5) == (
            labels_fp32 >= 0.5
        )
        clean_downweighted = clean_disagreement > threshold
        augmented_downweighted = augmented_disagreement > threshold

    temperature = float(raw["temperature"])
    if calibration_method == "per_view_affine_platt":
        clean_kd = binary_probability_distillation_loss(
            clean_output["logits"], clean_probability, clean_weights
        )
        augmented_kd = binary_probability_distillation_loss(
            augmented_output["logits"], augmented_probability, augmented_weights
        )
    else:
        clean_kd = binary_logit_distillation_loss(
            clean_output["logits"],
            probability_to_logit(clean_probability),
            clean_weights,
            temperature=temperature,
        )
        augmented_kd = binary_logit_distillation_loss(
            augmented_output["logits"],
            probability_to_logit(augmented_probability),
            augmented_weights,
            temperature=temperature,
        )
    teacher_loss = 0.5 * (clean_kd + augmented_kd)
    consistency = symmetric_bernoulli_kl(
        clean_output["logits"], augmented_output["logits"]
    )
    feature_loss = torch.zeros((), device=labels.device, dtype=torch.float32)
    pointwise_feature_loss = torch.zeros_like(feature_loss)
    relational_feature_loss = torch.zeros_like(feature_loss)
    quality_gate_loss = torch.zeros_like(feature_loss)
    feature_config = dict(raw.get("feature_distillation", {}))
    if bool(feature_config.get("enabled", False)):
        branch_weights = feature_config["branch_weights"]
        output_keys = {
            "semantic": "semantic_features",
            "forensic": "forensic_features",
            "fused": "features",
        }
        branch_losses: list[torch.Tensor] = []
        for branch_name, output_key in output_keys.items():
            if output_key not in clean_output or output_key not in augmented_output:
                raise KeyError(f"Student output is missing {output_key}")
            clean_feature_loss = normalized_feature_distillation_loss(
                clean_output[output_key],
                batch[f"teacher_m3_{branch_name}_clean"],
            )
            augmented_feature_loss = normalized_feature_distillation_loss(
                augmented_output[output_key],
                batch[f"teacher_m3_{branch_name}_aug"],
            )
            branch_losses.append(
                float(branch_weights[branch_name])
                * 0.5
                * (clean_feature_loss + augmented_feature_loss)
            )
        pointwise_feature_loss = sum(branch_losses)

        component_weights = dict(
            feature_config.get("component_weights", {"pointwise": 1.0})
        )
        relational_config = dict(feature_config.get("relational", {}))
        if bool(relational_config.get("enabled", False)):
            relational_branches = dict(relational_config["branch_weights"])
            pairwise_losses: list[torch.Tensor] = []
            cross_view_losses: list[torch.Tensor] = []
            for branch_name, branch_weight in relational_branches.items():
                output_key = output_keys[branch_name]
                clean_student = clean_output[output_key]
                augmented_student = augmented_output[output_key]
                clean_teacher = batch[f"teacher_m3_{branch_name}_clean"]
                augmented_teacher = batch[f"teacher_m3_{branch_name}_aug"]
                pairwise_losses.append(
                    float(branch_weight)
                    * 0.5
                    * (
                        relational_feature_distillation_loss(
                            clean_student, clean_teacher
                        )
                        + relational_feature_distillation_loss(
                            augmented_student, augmented_teacher
                        )
                    )
                )
                cross_view_losses.append(
                    float(branch_weight)
                    * cross_view_feature_delta_loss(
                        clean_student,
                        augmented_student,
                        clean_teacher,
                        augmented_teacher,
                    )
                )
            cross_view_weight = float(
                relational_config.get("cross_view_delta_weight", 0.0)
            )
            relational_feature_loss = (
                (1.0 - cross_view_weight) * sum(pairwise_losses)
                + cross_view_weight * sum(cross_view_losses)
            )

        gate_config = dict(feature_config.get("quality_gate_distillation", {}))
        if bool(gate_config.get("enabled", False)):
            required_gate_keys = {
                "student_clean": (clean_output, "gate_fractions"),
                "student_aug": (augmented_output, "gate_fractions"),
                "teacher_clean": (batch, "teacher_m3_gate_clean"),
                "teacher_aug": (batch, "teacher_m3_gate_aug"),
            }
            missing_gate_keys = [
                name
                for name, (container, key) in required_gate_keys.items()
                if key not in container
            ]
            if missing_gate_keys:
                raise KeyError(
                    f"Quality-gate distillation inputs are missing: {missing_gate_keys}"
                )
            quality_gate_loss = 0.5 * (
                F.mse_loss(
                    clean_output["gate_fractions"].float(),
                    batch["teacher_m3_gate_clean"].float(),
                )
                + F.mse_loss(
                    augmented_output["gate_fractions"].float(),
                    batch["teacher_m3_gate_aug"].float(),
                )
            )

        feature_loss = (
            float(component_weights.get("pointwise", 0.0))
            * pointwise_feature_loss
            + float(component_weights.get("relational", 0.0))
            * relational_feature_loss
            + float(component_weights.get("quality_gate", 0.0))
            * quality_gate_loss
        )

    warmup_epochs = int(raw.get("distillation_warmup_epochs", 0))
    if warmup_epochs > 0:
        if epoch is None:
            raise ValueError("Distillation warm-up requires the current epoch")
        distillation_ramp = min(1.0, max(0.0, float(epoch + 1) / warmup_epochs))
    else:
        distillation_ramp = 1.0
    total = float(raw["hard_label_weight"]) * hard_loss + distillation_ramp * (
        float(raw["soft_teacher_weight"]) * teacher_loss
        + float(raw["consistency_weight"]) * consistency
        + float(raw.get("feature_weight", 0.0)) * feature_loss
    )
    return total, {
        "hard_label": float(hard_loss.detach()),
        "soft_teacher": float(teacher_loss.detach()),
        "consistency": float(consistency.detach()),
        "feature": float(feature_loss.detach()),
        "feature_pointwise": float(pointwise_feature_loss.detach()),
        "feature_relational": float(relational_feature_loss.detach()),
        "quality_gate": float(quality_gate_loss.detach()),
        "distillation_ramp": float(distillation_ramp),
        "mean_teacher_reliability": float(
            (0.5 * (clean_weights.mean() + augmented_weights.mean())).detach()
        ),
        "teacher_wrong_fraction": float(
            (
                1.0
                - 0.5
                * (
                    clean_teacher_correct.float().mean()
                    + augmented_teacher_correct.float().mean()
                )
            ).detach()
        ),
        "teacher_positive_fraction": float(
            (
                0.5
                * (
                    (clean_probability >= 0.5).float().mean()
                    + (augmented_probability >= 0.5).float().mean()
                )
            ).detach()
        ),
        "mean_teacher_disagreement": float(
            (0.5 * (clean_disagreement.mean() + augmented_disagreement.mean()))
            .detach()
            .cpu()
        ),
        "downweighted_fraction": float(
            (
                0.5
                * (
                    clean_downweighted.float().mean()
                    + augmented_downweighted.float().mean()
                )
            )
            .detach()
            .cpu()
        ),
        "total": float(total.detach()),
    }


def load_calibration_temperatures(path: str | Path) -> dict[str, float]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    temperatures = payload["temperatures"]
    return {"m2": float(temperatures["m2"]), "m3": float(temperatures["m3"])}
