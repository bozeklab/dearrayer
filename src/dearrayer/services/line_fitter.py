import math
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression

Line = tuple[Literal["vertical"] | float, float]
Point = tuple[float, float]


class LineFitter:
    @staticmethod
    def fit_single_point(point: Point, angle_deg: float) -> Line:
        """
        Create a line through a single point at the specified angle.

        Args:
            point: (x, y) coordinates
            angle_deg: Line angle in degrees (0° = horizontal, 90° = vertical)

        Returns:
            (slope, y_intercept) or ("vertical", x_coordinate)
        """
        x, y = point

        if math.isclose(math.cos(math.radians(angle_deg)), 0.0, abs_tol=1e-6):
            return "vertical", x

        slope = math.tan(math.radians(angle_deg))
        intercept = y - slope * x
        return slope, intercept

    @staticmethod
    def fit_orthogonal(points: list[Point]) -> Line:
        """
        Fit line using orthogonal regression (minimizes perpendicular distances).
        Best for geometric line fitting where both x and y have similar scales.

        Args:
            points: List of (x, y) coordinates (must have 2+ points)

        Returns:
            (slope, y_intercept) or ("vertical", x_coordinate)
        """
        if len(points) < 2:
            raise ValueError("Need at least 2 points for orthogonal regression")

        # Convert to numpy and center the data
        points_array = np.array(points)
        centroid = np.mean(points_array, axis=0)
        centered = points_array - centroid

        # SVD to find principal component (line direction)
        Vt = cast(
            NDArray[np.float64],
            np.linalg.svd(centered, full_matrices=False)[2],
        )
        direction = Vt[0]  # [dx, dy]

        # Check for vertical line
        if abs(direction[0]) < 1e-10:
            return "vertical", centroid[0]

        # Calculate slope and intercept
        slope = direction[1] / direction[0]
        intercept = centroid[1] - slope * centroid[0]
        return slope, intercept

    @staticmethod
    def fit_least_squares(points: list[Point]) -> Line:
        """
        Fit line using ordinary least squares (minimizes vertical distances).
        Good when x is clearly the independent variable.

        Args:
            points: List of (x, y) coordinates (must have 2+ points)

        Returns:
            (slope, y_intercept) or ("vertical", x_coordinate)
        """
        if len(points) < 2:
            raise ValueError("Need at least 2 points for least squares")

        points_array = np.array(points)
        x_vals = points_array[:, 0]
        y_vals = points_array[:, 1]

        # Check for vertical line
        if np.allclose(x_vals, x_vals[0], atol=1e-10):
            return "vertical", float(x_vals[0])

        # Fit using sklearn
        X = x_vals.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, y_vals)  # pyright: ignore[reportUnknownMemberType]
        return float(model.coef_[0]), float(model.intercept_)
