import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Read the image in grayscale
image_path = "D:\\Purple Heart High Quality.jfif"  # Provide the correct path to your image
gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Step 2: Apply a threshold to convert the grayscale image into a binary image
# The threshold value can be adjusted. Everything below the threshold will be black, everything above will be white.
_, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

# Step 3: Display the original and binary images using matplotlib
plt.figure(figsize=(10,5))

# Display the original grayscale image
plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

# Display the binary image

plt.subplot(1, 2, 2)
plt.imshow(binary_image, cmap='gray')
plt.title('Binary Image (Thresholded)')
plt.axis('off')

plt.show()

# Step 4: Optionally, use connected components or contour detection for segmentation
# Find contours from the binary image (useful for object segmentation)
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image for visualization
segmented_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for colored output
cv2.drawContours(segmented_image, contours, -1, (0, 255, 0), 2)

# Show the segmented image with contours
plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
plt.title('Segmented Image with Contours')
plt.axis('off')
plt.show()
