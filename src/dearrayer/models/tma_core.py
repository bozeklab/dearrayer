from abc import ABC, abstractmethod
from dataclasses import dataclass


class TMACore(ABC):
    """Abstract base class for all TMA core types"""

    @property
    @abstractmethod
    def position(self) -> tuple[float, float]:
        """Core center coordinates (x, y)"""
        pass

    @property
    @abstractmethod
    def bounding_box(self) -> tuple[float, float, float, float]:
        """(x, y, width, height) in image coordinates"""
        pass

    @property
    @abstractmethod
    def is_detected(self) -> bool:
        """Whether core was physically detected"""
        pass


@dataclass
class DetectedTMACore(TMACore):
    """Concrete detected core with measurement data"""

    x: float
    y: float
    diameter: float
    name: str
    row_index: int
    col_index: int

    @property
    def position(self) -> tuple[float, float]:
        return (self.x, self.y)

    @property
    def bounding_box(self) -> tuple[float, float, float, float]:
        return (
            self.x - self.diameter / 2,
            self.y - self.diameter / 2,
            self.diameter,
            self.diameter,
        )

    @property
    def is_detected(self) -> bool:
        return True


@dataclass
class PredictedTMACore(TMACore):
    """Concrete predicted core position"""

    x: float
    y: float
    diameter: float
    name: str
    row_index: int
    col_index: int
    confidence: float = 1.0

    @property
    def position(self) -> tuple[float, float]:
        return (self.x, self.y)

    @property
    def bounding_box(self) -> tuple[float, float, float, float]:
        return (
            self.x - self.diameter / 2,
            self.y - self.diameter / 2,
            self.diameter,
            self.diameter,
        )

    @property
    def is_detected(self) -> bool:
        return False
