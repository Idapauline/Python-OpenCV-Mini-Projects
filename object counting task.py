import cv2
import numpy as np

# Load the image
image_path = "C:\\Users\\jgmid\\Downloads\\Car Traffic Light.jpg"
image = cv2.imread("C:\\Users\\jgmid\\Downloads\\Car Traffic Light.jpg")  # Correctly load the image using OpenCV

# Check if the image is loaded correctly
if image is None:
    print("Error: Could not load image.")
    exit()

# Preprocess the image (convert to grayscale and apply Gaussian blur)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
blurred = cv2.GaussianBlur(gray, (5, 5), 0)     # Apply Gaussian blur

# Apply binary thresholding
_, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)

# Find contours in the thresholded image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Minimum area for a contour to be considered an object
min_contour_area = 500

object_count = 0

# Iterate over contours and count objects
for contour in contours:
    if cv2.contourArea(contour) < min_contour_area:
        continue

    # Bounding box for each object
    (x, y, w, h) = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    object_count += 1

# Display the object count on the image
cv2.putText(image, f'Object Count: {object_count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Show the result
cv2.imshow("Object Counting", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
