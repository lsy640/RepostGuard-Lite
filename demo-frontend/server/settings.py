from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEMO_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = DEMO_ROOT.parent
DIST_ROOT = DEMO_ROOT / "dist"

MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 50_000_000
MAX_BATCH_FILES = 100
IMAGE_CACHE_SIZE = 16
JOB_TTL_SECONDS = 60 * 60
# Pillow reports multi-picture JPEG files as ``MPO`` even when they use a
# normal .jpg/.jpeg filename. Treat MPO as a JPEG container and analyse its
# first frame so camera and iPhone photos remain inside the advertised JPEG
# upload contract.
SUPPORTED_FORMATS = {"JPEG", "MPO", "PNG", "WEBP", "BMP", "TIFF", "GIF"}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    checkpoint: Path
    checkpoint_sha256: str
    config_sha256: str
    threshold: float
    clean_auroc: float


MODEL_SPECS = {
    "m2": ModelSpec(
        name="m2",
        checkpoint=REPOSITORY_ROOT / "outputs/community_forensics_v3/m2/best.pt",
        checkpoint_sha256="468d3a58603fdf8dfe1b234a24fd8e52a99c6e4881e921bef6bb0cea64bbac34",
        config_sha256="f0bf20ef8d5d193967d9588759b2bd2aa0b6a778741635ca5bdaddf40ef035d4",
        threshold=0.99658203125,
        clean_auroc=0.978116,
    ),
    "m3": ModelSpec(
        name="m3",
        checkpoint=REPOSITORY_ROOT / "outputs/community_forensics_v3/m3/best.pt",
        checkpoint_sha256="c83f70641a9c8d7f6808e794cfc8c28c0e478feeca7506e489c772a512115b2f",
        config_sha256="1173f7dd3908656f8fda4a292470880b3487c33f04bff466c0a57929e88aaed2",
        threshold=0.9970703125,
        clean_auroc=0.9782205,
    ),
}

CALIBRATION_PATH = (
    REPOSITORY_ROOT
    / "student_distillation/v3_first_m2_0_m3_100/teacher_calibration.json"
)
