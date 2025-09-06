import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Load the image
img = cv.imread(r"C:\Users\jgmid\OneDrive\Pictures\cartoon-little-star1[1].webp")
assert img is not None, "File could not be read, check the file path."

# Create a mask initialized as background (0) for all pixels
mask = np.zeros(img.shape[:2], np.uint8)

# Create temporary arrays for GrabCut algorithm
bgdModel = np.zeros((1, 65), np.float64)  # Background model
fgdModel = np.zeros((1, 65), np.float64)  # Foreground model

# Define a rectangle around the object (adjust the values to fit your image)
rect = (50, 50, img.shape[1] - 100, img.shape[0] - 100)

# Apply the GrabCut algorithm
cv.grabCut(img, mask, rect, bgdModel, fgdModel, 10, cv.GC_INIT_WITH_RECT)  # Increased iterations to 10 for better accuracy

# Create a refined mask by marking background and foreground areas
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# Segment the image using the refined mask
segmented_img = img * mask2[:, :, np.newaxis]

# Display the original and segmented images side by side for comparison
plt.figure(figsize=(10, 5))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# Segmented Image
plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(segmented_img, cv.COLOR_BGR2RGB))
plt.title('Segmented Image (GrabCut)')
plt.axis('off')

plt.show()