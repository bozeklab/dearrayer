from abc import ABC
from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Position:
    x: float
    y: float

    def rotate(self, angle_deg: float, clip: bool = False) -> "Position":
        """Immutable rotation around origin (clockwise for positive angles)"""
        angle_rad = -np.deg2rad(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        x = self.x * cos_a - self.y * sin_a
        y = self.x * sin_a + self.y * cos_a
        if clip:
            x, y = min(1, max(0, x)), min(1, max(0, y))
        return Position(x, y)

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True)
class TMACore(ABC):
    """Base class for all TMA cores"""

    position: Position
    diameter: float

    @property
    def bounding_box(self) -> tuple[float, float, float, float]:
        return (
            self.position.x - self.diameter / 2,
            self.position.y - self.diameter / 2,
            self.diameter,
            self.diameter,
        )


@dataclass(frozen=True)
class DetectedTMACore(TMACore):
    @property
    def is_detected(self) -> bool:
        return True


@dataclass(frozen=True)
class PredictedTMACore(TMACore):
    confidence: float = 1.0

    @property
    def is_detected(self) -> bool:
        return False

    def enlarge(self, scale_pc: float = 105.0) -> "PredictedTMACore":
        return replace(self, diameter=self.diameter * scale_pc / 100)
