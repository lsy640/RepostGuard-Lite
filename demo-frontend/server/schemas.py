from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ModelName = Literal["m2", "m3"]


class JpegControl(BaseModel):
    enabled: bool = False
    quality: int = Field(default=70, ge=30, le=100)


class BlurControl(BaseModel):
    enabled: bool = False
    sigma: float = Field(default=1.0, ge=0.0, le=2.5)


class ResizeControl(BaseModel):
    enabled: bool = False
    scale: float = Field(default=0.5, ge=0.25, le=1.0)


class NoiseControl(BaseModel):
    enabled: bool = False
    sigma: float = Field(default=0.02, ge=0.0, le=0.10)


class JitterControl(BaseModel):
    enabled: bool = False
    brightness: float = Field(default=1.2, ge=0.8, le=1.2)
    contrast: float = Field(default=1.2, ge=0.8, le=1.2)
    saturation: float = Field(default=1.2, ge=0.8, le=1.2)


class CropControl(BaseModel):
    enabled: bool = False
    ratio: float = Field(default=0.8, ge=0.75, le=1.0)


class RobustnessRequest(BaseModel):
    model: ModelName
    jpeg: JpegControl = Field(default_factory=JpegControl)
    blur: BlurControl = Field(default_factory=BlurControl)
    resize: ResizeControl = Field(default_factory=ResizeControl)
    noise: NoiseControl = Field(default_factory=NoiseControl)
    jitter: JitterControl = Field(default_factory=JitterControl)
    crop: CropControl = Field(default_factory=CropControl)
