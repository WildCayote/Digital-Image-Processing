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

def process_and_display(image_path, title="Original Image"):
    """Process and display an image with all transformations."""
    # Read the image
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    
    # Create figure with subplots
    plt.figure(figsize=(20, 10))
    
    # Original images
    plt.subplot(3, 4, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original RGB")
    plt.axis('off')
    
    plt.subplot(3, 4, 2)
    plt.imshow(gray, cmap='gray')
    plt.title("Original Grayscale")
    plt.axis('off')
    
    # Image Negative
    plt.subplot(3, 4, 3)
    plt.imshow(image_negative(original))
    plt.title("Image Negative (RGB)")
    plt.axis('off')
    
    plt.subplot(3, 4, 4)
    plt.imshow(image_negative(gray), cmap='gray')
    plt.title("Image Negative (Grayscale)")
    plt.axis('off')
    
    # Gamma Correction
    plt.subplot(3, 4, 5)
    plt.imshow(gamma_correction(original, 2.2))
    plt.title("Gamma Correction (RGB)")
    plt.axis('off')
    
    plt.subplot(3, 4, 6)
    plt.imshow(gamma_correction(gray, 2.2), cmap='gray')
    plt.title("Gamma Correction (Grayscale)")
    plt.axis('off')
    
    # Logarithmic Transform
    plt.subplot(3, 4, 7)
    plt.imshow(logarithmic_transform(original))
    plt.title("Logarithmic Transform (RGB)")
    plt.axis('off')
    
    plt.subplot(3, 4, 8)
    plt.imshow(logarithmic_transform(gray), cmap='gray')
    plt.title("Logarithmic Transform (Grayscale)")
    plt.axis('off')
    
    # Contrast Stretching
    plt.subplot(3, 4, 9)
    plt.imshow(contrast_stretching(original))
    plt.title("Contrast Stretching (RGB)")
    plt.axis('off')
    
    plt.subplot(3, 4, 10)
    plt.imshow(contrast_stretching(gray), cmap='gray')
    plt.title("Contrast Stretching (Grayscale)")
    plt.axis('off')
    
    # Histogram Equalization
    plt.subplot(3, 4, 11)
    plt.imshow(histogram_equalization(original))
    plt.title("Histogram Equalization (RGB)")
    plt.axis('off')
    
    plt.subplot(3, 4, 12)
    plt.imshow(histogram_equalization(gray), cmap='gray')
    plt.title("Histogram Equalization (Grayscale)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Create a new figure for the remaining transformations
    plt.figure(figsize=(20, 10))
    
    # Intensity Level Slicing
    plt.subplot(2, 3, 1)
    plt.imshow(intensity_level_slicing(original, 100, 200))
    plt.title("Intensity Level Slicing (RGB)")
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(intensity_level_slicing(gray, 100, 200), cmap='gray')
    plt.title("Intensity Level Slicing (Grayscale)")
    plt.axis('off')
    
    # Bit Plane Slicing
    plt.subplot(2, 3, 3)
    plt.imshow(bit_plane_slicing(original, 7))
    plt.title("Bit Plane 7 (RGB)")
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(bit_plane_slicing(gray, 7), cmap='gray')
    plt.title("Bit Plane 7 (Grayscale)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_path = "images/test.jpg"
    try:
        process_and_display(image_path)
    except Exception as e:
        print(f"Error: {e}") 