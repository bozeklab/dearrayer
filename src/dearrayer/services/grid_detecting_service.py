import math
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Literal, Protocol, TypeVar, cast, final

import cv2
import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray
from scipy import ndimage
from skimage import feature, filters, measure, transform, util
# fmt: off
from skimage.morphology import \
    convex_hull_object  # pyright: ignore[reportUnknownVariableType]
# fmt: on
from sklearn.cluster import KMeans

from dearrayer.models.tissue_microarray import TissueMicroarray
from dearrayer.models.tma_core import (DetectedTMACore, Position,
                                       PredictedTMACore, TMACore)
from dearrayer.models.tma_grid import GridCell, TMACorePredictor, TMAGrid
from dearrayer.services.line_fitter import Line, LineFitter, Point


class RegionProp(Protocol):
    area: float
    perimeter: float
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]
    label: int


@dataclass
class GridDetectingServiceParameters:
    core_diameter: int
    column_labels: list[str]
    row_labels: list[str]
    minimum_area: float = 0.25
    use_convex_hull: bool = True
    random_state: int = 42
    manual_threshold: int | None = None


AnyTMACore = TypeVar("AnyTMACore", bound=TMACore)


@final
class GridDetectingService:
    def __init__(self) -> None:
        pass

    def __call__(
        self,
        tma: TissueMicroarray,
        parameters: GridDetectingServiceParameters,
    ) -> TMAGrid:
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
            small_img,
            relative_core_diameter,
            use_convex_hull=parameters.use_convex_hull,
            manual_threshold=parameters.manual_threshold,
        )
        n_columns = len(parameters.column_labels)
        n_rows = len(parameters.row_labels)

        known_cores, predictor = GridDetectingService.detect_tma_cores(
            binary,
            relative_core_diameter,
            n_columns,
            n_rows,
            parameters.column_labels,
            parameters.row_labels,
            parameters.minimum_area,
            parameters.random_state
        )
        return TMAGrid(tma, known_cores, predictor)

    @staticmethod
    def make_binary_image(
        gray_image: MatLike,
        relative_core_diameter: float,
        use_convex_hull: bool,
        manual_threshold: int | None,
    ) -> MatLike:
        core_diameter = relative_core_diameter * max(gray_image.shape)
        kernel_size = int(core_diameter * 0.6 * 2) | 1
        background = cv2.morphologyEx(
            gray_image,
            cv2.MORPH_OPEN,
            np.ones((kernel_size, kernel_size), np.uint8),
        )
        img_sub = cv2.subtract(gray_image, background)
        if manual_threshold is not None:
            thresh = manual_threshold
        else:
            # fmt: off
            thresh = cast(int, filters.threshold_triangle(img_sub))  # pyright: ignore[reportUnknownMemberType]
            # fmt:on
        binary = (img_sub > thresh).astype(np.uint8) * 255

        clean_size = max(1, int(core_diameter * 0.02))
        kernel = np.ones((clean_size, clean_size), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = ndimage.binary_fill_holes(binary).astype(np.uint8) * 255
        if use_convex_hull:
            binary = convex_hull_object(binary) * 255
        return util.img_as_ubyte(binary)

    @staticmethod
    def detect_tma_cores(
        binary_image: MatLike,
        relative_core_diameter: float,
        n_columns: int,
        n_rows: int,
        grid_cell_col_labels: list[str],
        grid_cell_row_labels: list[str],
        min_area_relative: float,
        random_state:int,
    ) -> tuple[dict[GridCell, DetectedTMACore], TMACorePredictor]:
        labels = cast(
            NDArray[np.uint32],
            measure.label(  # pyright: ignore[reportUnknownMemberType]
                binary_image > 0
            ),
        )
        regions: list[RegionProp] = cast(
            list[RegionProp],
            measure.regionprops(  # pyright: ignore[reportUnknownMemberType]
                labels
            ),
        )

        max_dimension = max(binary_image.shape)
        core_diameter = relative_core_diameter * max_dimension

        min_area = (
            min_area_relative
            * np.pi
            * (core_diameter / 2)
            * (core_diameter / 2)
        )
        centroids: list[DetectedTMACore] = []

        for region in regions:
            if min_area < region.area:
                circles = GridDetectingService.get_centers_from_region(
                    labels, region, core_diameter / 2
                )
                for x, y in circles:
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
        row_kmeans = KMeans(n_clusters=n_rows, n_init=10, random_state=random_state)
        row_labels = (
            row_kmeans.fit_predict(  # pyright: ignore[reportUnknownMemberType]
                rotated_positions[:, 1].reshape(-1, 1)
            )
        )
        col_kmeans = KMeans(n_clusters=n_columns, n_init=10, random_state=random_state)
        col_labels = (
            col_kmeans.fit_predict(  # pyright: ignore[reportUnknownMemberType]
                rotated_positions[:, 0].reshape(-1, 1)
            )
        )
        row_order = np.argsort(row_kmeans.cluster_centers_[:, 0])
        row_idx_map = {old: new for new, old in enumerate(row_order)}
        col_order = np.argsort(col_kmeans.cluster_centers_[:, 0])
        col_idx_map = {old: new for new, old in enumerate(col_order)}
        return_dictionary: dict[GridCell, DetectedTMACore] = dict()

        all_cores_in_row: dict[int, list[tuple[float, float]]] = defaultdict(
            list
        )
        all_cores_in_col: dict[int, list[tuple[float, float]]] = defaultdict(
            list
        )
        for i, rotated_core in enumerate(rotated_cores):
            row_idx = row_idx_map[row_labels[i]]
            col_idx = col_idx_map[col_labels[i]]
            grid_cell = GridCell(
                grid_cell_col_labels[col_idx], grid_cell_row_labels[row_idx]
            )
            original_core = map_rotated_to_original[rotated_core]
            return_dictionary[grid_cell] = original_core
            all_cores_in_col[col_idx].append(original_core.position.as_tuple())
            all_cores_in_row[row_idx].append(original_core.position.as_tuple())

        column_lines = GridDetectingService.fit_lines(
            all_cores_in_col, angle - 90
        )
        row_lines = GridDetectingService.fit_lines(all_cores_in_row, angle)

        def predictor(label: GridCell) -> PredictedTMACore | None:
            try:
                row_idx = grid_cell_row_labels.index(label.row_label)
                col_idx = grid_cell_col_labels.index(label.col_label)
            except ValueError:
                return None

            col_line = column_lines.get(col_idx)
            row_line = row_lines.get(row_idx)
            if col_line is None or row_line is None:
                return None
            position = GridDetectingService.find_intersection(
                col_line, row_line
            )
            if position is None:
                return None

            return PredictedTMACore(
                position=position,
                diameter=relative_core_diameter,
            )

        return return_dictionary, predictor

    @staticmethod
    def fit_lines(
        point_groups: dict[int, list[Point]],
        fallback_angle_deg: float,
        method: str = "orthogonal",
    ) -> dict[int, Line]:
        """
        Fit lines to groups of points with fallback for single points.

        This is the main function you should use - it's a direct replacement
        for your original fit_line function.

        Args:
            point_groups: Dictionary mapping group IDs to lists of (x, y) points
            fallback_angle_deg: Angle for lines through single points
            method: "orthogonal" (recommended) or "least_squares"

        Returns:
            Dictionary mapping group IDs to fitted lines
        """
        if method not in ["orthogonal", "least_squares"]:
            raise ValueError(f"Unknown method: {method}")

        lines: dict[int, Line] = {}
        fitter = LineFitter()

        for group_id, points in point_groups.items():
            if not points:
                continue

            if len(points) == 1:
                # Single point: use the fallback angle
                lines[group_id] = fitter.fit_single_point(
                    points[0], fallback_angle_deg
                )
            else:
                # Multiple points: use specified method
                if method == "orthogonal":
                    lines[group_id] = fitter.fit_orthogonal(points)
                else:  # least_squares
                    lines[group_id] = fitter.fit_least_squares(points)

        return lines

    @staticmethod
    def find_intersection(
        line1: tuple[float | Literal["vertical"], float],
        line2: tuple[float | Literal["vertical"], float],
    ) -> Position | None:
        match (line1[0], line2[0]):
            case ("vertical", "vertical"):
                return None
            case ("vertical", m2):
                x = line1[1]
                c2 = line2[1]
                y = m2 * x + c2
            case (m1, "vertical"):
                x = line2[1]
                c1 = line1[1]
                y = m1 * x + c1
            case (m1, m2):
                c1 = line1[1]
                c2 = line2[1]
                if math.isclose(m1, m2, abs_tol=1e-9):
                    return None
                x = (c2 - c1) / (m1 - m2)
                y = m1 * x + c1
        return Position(x, y)

    @staticmethod
    def estimate_dominant_grid_angle(centroids: list[DetectedTMACore]) -> float:
        angles: list[float] = []
        n = len(centroids)
        for i in range(n):
            for j in range(i + 1, n):
                dx = centroids[j].position.x - centroids[i].position.x
                dy = centroids[j].position.y - centroids[i].position.y
                angle = math.atan2(dy, dx)
                angle_deg = cast(float,np.rad2deg(angle))

                # This is the key change: map to [-45, 45] instead of [-90, 90]
                while angle_deg > 45:
                    angle_deg -= 90
                while angle_deg <= -45:
                    angle_deg += 90

                angles.append(angle_deg)

        hist, bin_edges = np.histogram(angles, bins=90, range=(-45, 45))
        dominant_angle = cast(float,bin_edges[np.argmax(hist)])
        return dominant_angle

    @staticmethod
    def rotate_centroids(
        centroids: list[AnyTMACore], angle_deg: float
    ) -> list[AnyTMACore]:
        return [
            replace(c, position=c.position.rotate(angle_deg=angle_deg))
            for c in centroids
        ]

    @staticmethod
    def get_centers_from_region(
        label_img: MatLike,
        region: RegionProp,
        expected_radius: float,
        radius_margin: int = 5,
        sigma: float = 2,
        padding: int = 2,
    ) -> list[tuple[float, float]]:
        """
        Detect circle center in a single labeled region using Hough transform.

        Parameters:
            label_img (np.ndarray): Labeled image (same shape as original binary image).
            region (RegionPrope): Single region from skimage.measure.regionprops.
            expected_radius (float): Expected approximate radius of the circle.
            radius_margin (int): Search radii in range [expected_radius - margin, expected_radius + margin].
            sigma (float): Sigma for Canny edge detection.

        Returns:
            (center_x, center_y): Tuple of float with center coordinates in original image.
            Returns region centroid if detection fails.
        """
        minr, minc, maxr, maxc = region.bbox
        cropped = (
            label_img[
                max(0, minr - padding) : maxr + padding,
                max(0, minc - padding) : maxc + padding,
            ]
            == region.label
        )

        edges = cast(
            MatLike,
            feature.canny(  # pyright: ignore[reportUnknownMemberType]
                cropped, sigma=sigma
            ),
        )

        hough_radii = np.arange(
            max(1, int(expected_radius - radius_margin)),
            int(expected_radius + radius_margin + 1),
        )

        hough_res = cast(
            NDArray[np.float64],
            transform.hough_circle(  # pyright: ignore[reportUnknownMemberType]
                edges, hough_radii, full_output=True
            ),
        )

        accums, cx, cy, radii = cast(
            tuple[
                NDArray[np.float64],
                NDArray[np.int64],
                NDArray[np.int64],
                NDArray[np.int64],
            ],
            transform.hough_circle_peaks(  # pyright: ignore[reportUnknownMemberType]
                hough_res, hough_radii, total_num_peaks=10
            ),
        )
        largest_radius = np.max(hough_radii)
        cx = cx - largest_radius + minc
        cy = cy - largest_radius + minr
        confidences = accums / np.max(hough_res)
        threshold = 0.7
        valid = confidences > threshold

        valid_circles = list(zip(cx[valid], cy[valid], radii[valid]))
        distinct_circles: list[tuple[int, int, int]] = []

        for x1, y1, r1 in valid_circles:
            too_close = False
            for x2, y2, r2 in distinct_circles:
                dist = np.hypot(x1 - x2, y1 - y2)
                if dist < (r1 + r2) * 0.9:
                    too_close = True
                    break
            if not too_close:
                distinct_circles.append((x1, y1, r1))

        return [x[:2] for x in distinct_circles]
