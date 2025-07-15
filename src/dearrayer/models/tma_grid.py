from dataclasses import dataclass
from typing import Protocol

from dearrayer.models.tissue_microarray import TissueMicroarray
from dearrayer.models.tma_core import PredictedTMACore, TMACore


@dataclass
class GridCell:
    col_label: str
    row_label: str


class TMACorePredictor(Protocol):
    def predict(self, label: GridCell) -> PredictedTMACore | None:
        pass


class TMAGrid:
    def __init__(
        self,
        tma: TissueMicroarray,
        labels: list[GridCell],
        core_predictor: TMACorePredictor,
    ):
        self.tma = tma
        self.labels = labels
        self.core_predictor = core_predictor
        raise NotImplementedError()

    def get_or_predict(self, grid_cell_label: GridCell) -> TMACore:
        raise NotImplementedError()
