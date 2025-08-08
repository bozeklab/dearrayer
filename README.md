# Dearrayer

A Python tool for automated dearraying of Tissue Microarrays (TMA) with grid detection and core extraction capabilities.

## Overview

Dearrayer is a specialized image analysis tool designed to automatically detect and extract tissue cores from Tissue Microarray (TMA) images. It provides robust grid detection algorithms that can handle rotated arrays and predict missing cores based on the detected grid pattern.

## Features

- **Automatic Grid Detection**: Detects TMA grid structure even with rotation and missing cores
- **Core Detection**: Identifies tissue cores using advanced image processing techniques
- **Core Prediction**: Predicts locations of missing cores based on grid pattern
- **Flexible Grid Configuration**: Supports custom row and column labeling schemes
- **High-Resolution Support**: Handles large microscopy images with automatic downsampling
- **CZI Format Support**: Works with Carl Zeiss Image (CZI) files through pylibCZIrw integration

## Installation


```bash
# using uv:
uv add dearrayer

# or pip:
pip install dearrayer
```

### Optional: CZI File Support

If you want to work with Carl Zeiss CZI files, install the additional dependency:

```bash
# using uv:
uv add pylibCZIrw

# or pip
pip install pylibCZIrw
```

### Development Installation

```bash
git clone https://github.com/bozeklab/dearrayer.git
cd dearrayer
uv sync
```

## Quick Start

### Basic Usage

```python
from dearrayer.models import TissueMicroarray
from dearrayer.services import GridDetectingService, GridDetectingServiceParameters
from skimage.util import img_as_ubyte
import numpy as np

# Load your image (as uint8 2D array)
image = img_as_ubyte(your_image)
tma = TissueMicroarray(image)

# Define grid parameters
column_labels = ["12", "11", "10", "9", "8", "7", "6", "5", "4", "3", "2", "1"]
row_labels = ["A", "B", "C", "D", "E", "F", "G", "H"]

params = GridDetectingServiceParameters(
    core_diameter=100,  # Expected core diameter in pixels
    column_labels=column_labels,
    row_labels=row_labels,
    minimum_area=0.25  # Minimum relative area for core detection
)

# Detect grid
service = GridDetectingService()
grid = service(tma, params)

# Access detected cores
from dearrayer.models import GridCell
core = grid.get_or_predict(GridCell("8", "C"))
if core:
    print(f"Core position: ({core.position.x}, {core.position.y})")
    print(f"Core diameter: {core.diameter}")
    print(f"Was detected: {core.is_detected}")
```

### Working with CZI Files

```python
from pylibCZIrw import czi as pyczi
from dearrayer.models import TissueMicroarray, GridCell
from dearrayer.services import GridDetectingService, GridDetectingServiceParameters
from skimage.util import img_as_ubyte

def mm_to_pixels(value_mm, metadata_dict):
    """Convert millimeters to pixels using CZI metadata"""
    scaling = metadata_dict["ImageDocument"]['Metadata']["Scaling"]["Items"]["Distance"]
    x_scaling = next(item for item in scaling if item["@Id"].lower() == "x")
    value_µm = value_mm * 1000
    return value_µm / (float(x_scaling["Value"]) * 1E6)

# Define your grid layout
column_labels = [str(i) for i in range(12, 0, -1)] + ["-3", "-2", "-1"]
row_labels = [chr(ord("A") + i) for i in reversed(range(8))]

with pyczi.open_czi("your_tma_file.czi") as czidoc:
    # Calculate core size in pixels
    core_size_mm = 1.2
    px_core_size = mm_to_pixels(core_size_mm, czidoc.metadata)
    
    # Set desired processing size
    desired_core_px = 100
    downscale = px_core_size // desired_core_px
    
    # Extract image with downscaling
    img = czidoc.read(plane={"T": 0, "Z": 0, "C": 0}, zoom=1/downscale)
    tma = TissueMicroarray(img_as_ubyte(img.squeeze()))
    
    # Configure detection parameters
    params = GridDetectingServiceParameters(
        core_diameter=desired_core_px,
        column_labels=column_labels,
        row_labels=row_labels
    )
    
    # Detect grid
    service = GridDetectingService()
    grid = service(tma, params)
    
    # Extract specific core at full resolution
    core = grid.get_or_predict(GridCell("8", "C"))
    if core:
        # Calculate ROI in full resolution coordinates
        min_x, min_y, w, h = czidoc.total_bounding_rectangle
        max_size = max(w, h)
        roi = tuple(int(x * max_size) for x in core.bounding_box)
        roi = (min_x + roi[0], min_y + roi[1], roi[2], roi[3])
        
        # Extract full resolution core
        core_img = czidoc.read(roi=roi, plane={"T": 0, "Z": 0, "C": 1}, zoom=1)
```

## API Reference

### Core Classes

#### `TissueMicroarray`
Represents a tissue microarray image.
- `image`: 2D numpy array with dtype uint8
- `dimensions`: Returns image dimensions (height, width)

#### `TMACore` (Abstract Base)
Base class for all TMA cores.
- `position`: Core center position (normalized coordinates)
- `diameter`: Core diameter (normalized to max image dimension)
- `bounding_box`: Returns (x, y, width, height) tuple

#### `DetectedTMACore`
Represents a core that was automatically detected.
- `is_detected`: Always returns `True`

#### `PredictedTMACore`
Represents a core whose position was predicted based on grid pattern.
- `confidence`: Prediction confidence (0.0 to 1.0)
- `is_detected`: Always returns `False`
- `enlarge(scale_pc)`: Returns enlarged version of the core

#### `GridCell`
Represents a grid position.
- `col_label`: Column identifier
- `row_label`: Row identifier

#### `TMAGrid`
Manages the detected grid and provides core access.
- `get_or_predict(grid_cell)`: Returns detected core or prediction for given grid cell

### Services

#### `GridDetectingService`
Main service for detecting TMA grids.

#### `GridDetectingServiceParameters`
Configuration for grid detection:
- `core_diameter`: Expected core diameter in pixels.
- `column_labels`: List of column identifiers.
- `row_labels`: List of row identifiers.
- `minimum_area`: Minimum relative area for core detection (default: 0.25).
- `use_convex_hull`: Whether to use convex hull computation in the core detection process, takes significantly longer, but sometimes gives better results (default: True).
- `random_state`: Random state used for KMeans initialization (default: 42).
- `manual_threshold`: Option to manually set the binarization threshold (default: None - automatic thresholding. Can be int or None).
- `radius_margin`: Margin of radii search in Hough transform for core detection. Worth limiting when we get some artifacts detected instead of actual cores (default: 5).

## Algorithm Details

The grid detection algorithm works in several stages:

1. **Image Preprocessing**: Applies morphological operations and threshold detection
2. **Core Detection**: Identifies potential tissue cores using region analysis and Hough transforms
3. **Grid Estimation**: Estimates dominant grid orientation using pairwise angle analysis
4. **Clustering**: Groups cores into rows and columns using K-means clustering
5. **Line Fitting**: Fits lines through detected cores to enable prediction of missing cores

## Requirements

- Python ≥ 3.11
- numpy ≥ 2.3.1
- opencv-python ≥ 4.11.0.86
- scikit-image ≥ 0.25.2
- scikit-learn ≥ 1.7.0
- scipy ≥ 1.16.0
- matplotlib ≥ 3.10.3

### Optional Dependencies

- `pylibCZIrw`: For reading Carl Zeiss CZI files

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.


## Support

For questions, bug reports, or feature requests, please open an issue on GitHub.
