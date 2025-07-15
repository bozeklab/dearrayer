from dataclasses import dataclass
from typing import Protocol

from dearrayer.models.tissue_microarray import TissueMicroarray
from dearrayer.models.tma_core import DetectedTMACore, PredictedTMACore, TMACore


@dataclass(frozen=True)
class GridCell:
    col_label: str
    row_label: str


class TMACorePredictor(Protocol):
    def __call__(self, label: GridCell) -> PredictedTMACore | None:
        pass


class TMAGrid:
    def __init__(
        self,
        tma: TissueMicroarray,
        detected_cores: dict[GridCell, DetectedTMACore],
        core_predictor: TMACorePredictor,
    ):
        self.tma = tma
        self.detected_cores = detected_cores
        self.core_predictor = core_predictor

    def get_or_predict(self, grid_cell_label: GridCell) -> TMACore | None:
        return self.detected_cores.get(
            grid_cell_label, self.core_predictor(grid_cell_label)
        )
