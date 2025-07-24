import imageio.v3 as io
import matplotlib.pyplot as plt
from matplotlib import patches
from numpy import uint8
from numpy.typing import NDArray

from dearrayer.models import GridCell, TissueMicroarray, DetectedTMACore
from dearrayer.services import (
    GridDetectingService,
    GridDetectingServiceParameters,
)

if __name__ == "__main__":

    col_lab = [
        "12",
        "11",
        "10",
        "9",
        "8",
        "7",
        "6",
        "5",
        "4",
        "3",
        "2",
        "1",
        "-3",
        "-2",
        "-1",
    ]
    row_lab = ["H", "G", "F", "E", "D", "C", "B", "A"]
    gds = GridDetectingService()
    grid_detecting_parameters = GridDetectingServiceParameters(
        core_diameter=350, column_labels=col_lab, row_labels=row_lab
    )
    import pathlib as p

    paths = list(p.Path("../../notebooks").glob("Cycle*TMA_007.png"))

    for tma_image_path in paths:
        print(tma_image_path.stem)
        tma_img: NDArray[uint8] = (
            io.imread(  # pyright: ignore[reportUnknownMemberType]
                tma_image_path
            )
        )
        tma = TissueMicroarray(tma_img)
        col_lab = [
            "12",
            "11",
            "10",
            "9",
            "8",
            "7",
            "6",
            "5",
            "4",
            "3",
            "2",
            "1",
            "-3",
            "-2",
            "-1",
        ]
        row_lab = ["H", "G", "F", "E", "D", "C", "B", "A"]
        grid = gds(tma, grid_detecting_parameters)

        excluded_coords = {(a, b) for a in col_lab[-3:] for b in row_lab[:-2]}

        plt.show()  # pyright: ignore[reportUnknownMemberType]
        plt.imshow(  # pyright: ignore[reportUnknownMemberType]
            tma.image, cmap="gray"
        )
        plt.title(  # pyright: ignore[reportUnknownMemberType]
            f"{tma_image_path.stem}"
        )
        for coords in [(c, r)
                       for c in col_lab
                       for r in row_lab
                       if (c, r) not in excluded_coords]:
            ax = plt.gca()
            gc = GridCell(*coords)
            dg = grid.get_or_predict(gc)
            was_detected = isinstance(dg, DetectedTMACore)

            if dg is None:
                print(f"Couldn't find core for {gc}")
                continue
            xy = (
                dg.position.x * max(tma.image.shape),
                dg.position.y * max(tma.image.shape),
            )
            ax.add_patch(
                patches.Circle(
                    xy,
                    dg.diameter * max(tma.image.shape) / 2,
                    color="red" if was_detected else "pink",
                    alpha=0.3,
                )
            )
            ax.annotate(  # pyright: ignore[reportUnknownMemberType]
                "".join(coords),
                xy,
                fontsize=9,
                ha="center",
                va="center_baseline",
            )
        plt.axis("off")  # pyright: ignore[reportUnknownMemberType]
        plt.gcf().set_size_inches((12, 6))
        plt.show()  # pyright: ignore[reportUnknownMemberType]
