# Digital Image Processing - Elementary Methods

This project implements various elementary digital image processing methods in Python. The implementation includes both RGB and grayscale image processing capabilities.

## Implemented Methods

1. Image Negative
2. Gamma Encoding/Correction
3. Logarithmic Transform
4. Contrast Stretching
5. Histogram Equalization
6. Intensity Level Slicing
7. Bit Plane Slicing

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- Matplotlib

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Place your test images in the project directory
2. Modify the `image_path` variable in `image_processing.py` to point to your image
3. Run the script:

```bash
python image_processing.py
```

The script will display the original image and all the transformations applied to it, both for RGB and grayscale versions.

## Features

- Handles both RGB and grayscale images
- Visualizes all transformations using matplotlib
- Includes error handling for image loading and processing
- Provides clear visualization of before/after results

## Notes

- For best results, use images that clearly demonstrate the effects of each transformation
- The script will automatically convert RGB images to grayscale for grayscale processing
- All transformations are applied to both RGB and grayscale versions of the input image
