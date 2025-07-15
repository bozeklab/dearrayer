from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


class InvalidImageType(ValueError):
    """Raised when the image is not a valid uint8 2D array."""

    ...


@dataclass
class TissueMicroarray:
    """
    Represents a tissue microarray image.

    Attributes:
        image: A 2D numpy array with dtype uint8.
    """

    image: NDArray[np.uint8]

    def __post_init__(self):
        if self.image.ndim != 2:
            raise InvalidImageType(
                f"image must be 2D; got shape {self.image.shape}"
            )
        if self.image.dtype != np.uint8:
            raise InvalidImageType(
                f"image must have dtype uint8; got {self.image.dtype}"
            )

    @property
    def dimensions(self) -> tuple[int, int]:
        """Returns the image dimensions (height, width)."""
        return self.image.shape
