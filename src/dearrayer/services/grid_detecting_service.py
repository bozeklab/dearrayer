import math
from dataclasses import dataclass, replace
from typing import Protocol, TypeVar, cast

import cv2
import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray
from scipy import ndimage
from skimage import filters, measure
from sklearn.cluster import KMeans

from dearrayer.models.tissue_microarray import TissueMicroarray
from dearrayer.models.tma_core import (
    DetectedTMACore,
    Position,
    PredictedTMACore,
    TMACore,
)
from dearrayer.models.tma_grid import GridCell, TMACorePredictor, TMAGrid


class RegionProp(Protocol):
    area: float
    perimeter: float
    centroid: tuple[float, float]


@dataclass
class GridDetectingServiceParameters:
    core_diameter: int
    column_labels: list[str]
    row_labels: list[str]
    circularity_threshold: float = 0.6


AnyTMACore = TypeVar("AnyTMACore", bound=TMACore)


class GridDetectingService:
    def __init__(self) -> None:
        pass

    def __call__(
        self,
        tma: TissueMicroarray,
        parameters: GridDetectingServiceParameters,
    ) -> TMAGrid:  # tuple[dict[GridCell, DetectedTMACore], TMACorePredictor]:

        core_diameter = parameters.core_diameter
        height, width = tma.dimensions
        max_dim = max(width, height)
        relative_core_diameter = core_diameter / max_dim
        downsample = max(1, round(max_dim / 1200))
        small_img = cv2.resize(
            tma.image,
            (int(width / downsample), int(height / downsample)),
            interpolation=cv2.INTER_AREA,
        )
        binary = GridDetectingService.make_binary_image(
            small_img, relative_core_diameter
        )
        n_columns = len(parameters.column_labels)
        n_rows = len(parameters.row_labels)

        known_cores, predictor = GridDetectingService.detect_tma_cores(
            binary,
            relative_core_diameter,
            n_columns,
            n_rows,
            parameters.circularity_threshold,
            parameters.column_labels,
            parameters.row_labels,
        )
        return TMAGrid(tma, known_cores, predictor)

    @staticmethod
    def make_binary_image(
        gray_image: MatLike,
        relative_core_diameter: float,
    ) -> MatLike:
        core_diameter = relative_core_diameter * max(gray_image.shape)
        kernel_size = int(core_diameter * 0.6 * 2) | 1
        background = cv2.morphologyEx(
            gray_image,
            cv2.MORPH_OPEN,
            np.ones((kernel_size, kernel_size), np.uint8),
        )
        img_sub = cv2.subtract(gray_image, background)

        # fmt: off
        thresh = cast(float, filters.threshold_triangle(img_sub))  # pyright: ignore[reportUnknownMemberType]
        # fmt:on
        binary = (img_sub > thresh).astype(np.uint8) * 255

        clean_size = max(1, int(core_diameter * 0.02))
        kernel = np.ones((clean_size, clean_size), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = ndimage.binary_fill_holes(binary).astype(np.uint8) * 255
        return binary

    @staticmethod
    def detect_tma_cores(
        binary_image: MatLike,
        relative_core_diameter: float,
        n_columns: int,
        n_rows: int,
        circularity_threshold: float,
        grid_cell_col_labels: list[str],
        grid_cell_row_labels: list[str],
    ) -> tuple[dict[GridCell, DetectedTMACore], TMACorePredictor]:
        # fmt: off
        labels = cast(NDArray[np.uint32], measure.label(binary_image > 0))  # pyright: ignore[reportUnknownMemberType]
        regions: list[RegionProp]= cast(list[RegionProp], measure.regionprops(labels))  # pyright: ignore[reportUnknownMemberType]
        # fmt: on

        max_dimension = max(binary_image.shape)
        core_diameter = relative_core_diameter * max_dimension

        min_area = 0.5 * np.pi * (core_diameter / 2) ** 2
        max_area = 2.0 * np.pi * (core_diameter / 2) ** 2
        centroids: list[DetectedTMACore] = []

        for region in regions:
            if min_area < region.area < max_area:
                perimeter = region.perimeter
                circularity = (
                    4 * np.pi * region.area / (perimeter**2)
                    if perimeter > 0
                    else 0
                )
                if circularity > circularity_threshold:
                    x, y = region.centroid[::-1]
                    relative_x, relative_y = (
                        x / max_dimension,
                        y / max_dimension,
                    )
                    centroids.append(
                        DetectedTMACore(
                            position=Position(relative_x, relative_y),
                            diameter=relative_core_diameter,
                        )
                    )

        if not centroids:
            return {}, lambda label: None

        angle = GridDetectingService.estimate_dominant_grid_angle(centroids)
        rotated_cores = GridDetectingService.rotate_centroids(centroids, angle)

        map_rotated_to_original = {
            r: o for r, o in zip(rotated_cores, centroids)
        }

        rotated_positions = np.array(
            [
                [rotated_core.position.x, rotated_core.position.y]
                for rotated_core in rotated_cores
            ],
            np.float32,
        )
        row_kmeans = KMeans(n_clusters=n_rows, n_init=10)
        row_labels = (
            row_kmeans.fit_predict(  # pyright: ignore[reportUnknownMemberType]
                rotated_positions[:, 1].reshape(-1, 1)
            )
        )
        # Cluster x into columns
        col_kmeans = KMeans(n_clusters=n_columns, n_init=10)
        col_labels = (
            col_kmeans.fit_predict(  # pyright: ignore[reportUnknownMemberType]
                rotated_positions[:, 0].reshape(-1, 1)
            )
        )
        # Order rows top to bottom by cluster center y
        row_order = np.argsort(row_kmeans.cluster_centers_[:, 0])
        row_idx_map = {old: new for new, old in enumerate(row_order)}
        # Order columns left to right by cluster center x
        col_order = np.argsort(col_kmeans.cluster_centers_[:, 0])
        col_idx_map = {old: new for new, old in enumerate(col_order)}
        return_dictionary: dict[GridCell, DetectedTMACore] = dict()
        for i, rotated_core in enumerate(rotated_cores):
            x, y = rotated_core.position.x, rotated_core.position.y
            row_idx = row_idx_map[row_labels[i]]
            col_idx = col_idx_map[col_labels[i]]
            grid_cell = GridCell(
                grid_cell_col_labels[col_idx], grid_cell_row_labels[row_idx]
            )
            return_dictionary[grid_cell] = map_rotated_to_original[rotated_core]

        def predictor(label: GridCell) -> PredictedTMACore | None:
            try:
                row_idx = grid_cell_row_labels.index(label.row_label)
                col_idx = grid_cell_col_labels.index(label.col_label)
            except ValueError:
                return None  # Invalid label

            # Get cluster index from logical index (e.g. 'B' → 1)
            cluster_row_idx = row_order[row_idx]
            cluster_col_idx = col_order[col_idx]

            # Get rotated coordinates from cluster centers
            rotated_x = col_kmeans.cluster_centers_[cluster_col_idx][0]
            rotated_y = row_kmeans.cluster_centers_[cluster_row_idx][0]

            # Rotate back to original space
            rotated_position = Position(rotated_x, rotated_y)
            original_position = rotated_position.rotate(
                angle_deg=-angle, clip=True
            )

            return PredictedTMACore(
                position=original_position,
                diameter=relative_core_diameter,
            )

        return return_dictionary, predictor

    @staticmethod
    def estimate_dominant_grid_angle(centroids: list[DetectedTMACore]) -> float:
        angles: list[float] = []
        n = len(centroids)
        for i in range(n):
            for j in range(i + 1, n):
                dx = centroids[j].position.x - centroids[i].position.x
                dy = centroids[j].position.y - centroids[i].position.y
                angle = math.atan2(dy, dx)
                angle_deg = np.rad2deg(angle)
                if angle_deg < -90:
                    angle_deg += 180
                elif angle_deg > 90:
                    angle_deg -= 180
                angles.append(angle_deg)
        hist, bin_edges = np.histogram(angles, bins=180, range=(-90, 90))
        dominant_angle = bin_edges[np.argmax(hist)]
        return dominant_angle

    @staticmethod
    def rotate_centroids(
        centroids: list[AnyTMACore], angle_deg: float
    ) -> list[AnyTMACore]:
        return [
            replace(c, position=c.position.rotate(angle_deg=angle_deg))
            for c in centroids
        ]
