import cv2
import numpy as np
import matplotlib.pyplot as plt

def image_negative(image):
    """Compute the negative of an image."""
    return 255 - image

def gamma_correction(image, gamma):
    """Apply gamma correction to an image."""
    # Create a lookup table for gamma correction
    lookup_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, lookup_table)

def logarithmic_transform(image, c=1):
    """Apply logarithmic transform to an image."""
    # Add 1 to avoid log(0)
    return c * np.log(1 + image.astype(float))

def contrast_stretching(image):
    """Apply contrast stretching to an image."""
    if len(image.shape) == 3:  # RGB image
        result = np.zeros_like(image)
        for i in range(3):
            channel = image[:,:,i]
            min_val = np.min(channel)
            max_val = np.max(channel)
            result[:,:,i] = ((channel - min_val) / (max_val - min_val)) * 255
        return result.astype(np.uint8)
    else:  # Grayscale image
        min_val = np.min(image)
        max_val = np.max(image)
        return (((image - min_val) / (max_val - min_val)) * 255).astype(np.uint8)

def histogram_equalization(image):
    """Apply histogram equalization to an image."""
    if len(image.shape) == 3:  # RGB image
        # Convert to YUV color space
        yuv_img = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        # Apply histogram equalization to Y channel
        yuv_img[:,:,0] = cv2.equalizeHist(yuv_img[:,:,0])
        # Convert back to BGR
        return cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR)
    else:  # Grayscale image
        return cv2.equalizeHist(image)

def intensity_level_slicing(image, lower_bound, upper_bound, preserve_background=True):
    """Apply intensity level slicing to an image."""
    if len(image.shape) == 3:  # RGB image
        result = np.zeros_like(image)
        for i in range(3):
            channel = image[:,:,i]
            if preserve_background:
                result[:,:,i] = np.where((channel >= lower_bound) & (channel <= upper_bound), 255, channel)
            else:
                result[:,:,i] = np.where((channel >= lower_bound) & (channel <= upper_bound), 255, 0)
        return result
    else:  # Grayscale image
        if preserve_background:
            return np.where((image >= lower_bound) & (image <= upper_bound), 255, image)
        else:
            return np.where((image >= lower_bound) & (image <= upper_bound), 255, 0)

def bit_plane_slicing(image, bit_plane):
    """Extract a specific bit plane from an image."""
    if bit_plane < 0 or bit_plane > 7:
        raise ValueError("Bit plane must be between 0 and 7")
    
    # Get the bit plane
    return ((image >> bit_plane) & 1) * 255
