from typing import Any

from dearrayer.models.tissue_microarray import TissueMicroarray
from dearrayer.models.tma_core import DetectedTMACore
from dearrayer.models.tma_grid import GridCell, TMACorePredictor


class GridDetectingService:
    def __init__(self) -> None:
        pass

    def __call__(
        self, tma: TissueMicroarray, parameters: dict[str, Any]
    ) -> tuple[dict[GridCell, DetectedTMACore], TMACorePredictor]:

        raise NotImplementedError()
