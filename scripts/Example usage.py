import imageio.v3 as io
import matplotlib.pyplot as plt
from matplotlib import patches
from numpy import uint8
from numpy.typing import NDArray

from dearrayer.models import GridCell, TissueMicroarray
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

    paths = list(p.Path("images").glob("Cycle*TMA_007.png"))

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
        gds = GridDetectingService()
        grid = gds(tma, grid_detecting_parameters)
        plt.show()  # pyright: ignore[reportUnknownMemberType]
        plt.imshow(  # pyright: ignore[reportUnknownMemberType]
            tma.image, cmap="gray"
        )
        plt.title(  # pyright: ignore[reportUnknownMemberType]
            f"{tma_image_path.stem}"
        )
        for gc, dg in grid.detected_cores.items():
            xy = (
                dg.position.x * max(tma.image.shape),
                dg.position.y * max(tma.image.shape),
            )
            ax = plt.gca()
            ax.add_patch(
                patches.Circle(
                    xy,
                    dg.diameter * max(tma.image.shape) / 2,
                    color="red",
                    alpha=0.3,
                )
            )
            ax.annotate(  # pyright: ignore[reportUnknownMemberType]
                "".join((gc.col_label, gc.row_label)),
                xy,
                fontsize=9,
                ha="center",
                va="center_baseline",
            )

        for coords in [
            (c, r)
            for c in col_lab
            for r in row_lab
            if (
                GridCell(c, r) not in grid.detected_cores
                and (c, r)
                not in {(a, b) for a in col_lab[-3:] for b in row_lab[:-2]}
            )
        ]:
            ax = plt.gca()
            gc = GridCell(*coords)
            dg = grid.get_or_predict(gc)
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
                    color="pink",
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
