import cv2
import numpy as np

def detect_color(image):
    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for red, yellow, and green
    color_ranges = {
        "Red": [([0, 100, 100], [10, 255, 255]), ([170, 100, 100], [180, 255, 255])],  # Lower and Upper red
        "Yellow": ([20, 100, 100], [30, 255, 255]),
        "Green": ([40, 100, 100], [70, 255, 255])
    }

    detected_colors = []

    # Check for each color
    for color, ranges in color_ranges.items():
        for lower, upper in (ranges if isinstance(ranges[0], tuple) else [ranges]):
            # Create a mask for the specific color
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            # Calculate the percentage of the detected color in the image
            color_percentage = cv2.countNonZero(mask) / (image.shape[0] * image.shape[1])
            if color_percentage > 0.01:  # Adjust the threshold as necessary
                detected_colors.append(color)

    return detected_colors

# Load the input image for color detection
image_path =  "C:\\Users\\jgmid\\OneDrive\\Pictures\\Saved Pictures\\traff light.jpeg" # Update with your image path
image = cv2.imread(image_path)

# Check if the image was successfully loaded
if image is None:
    print("Error: Could not open or find the image.")
    exit()  # Stop further execution if image is not loaded

# Detect colors in the image
detected_colors = detect_color(image)

# Display the image with detected colors in a window
cv2.putText(image, f'Detected Colors: {", ".join(detected_colors) if detected_colors else "No Colors Detected"}',
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

cv2.imshow('Traffic Sign with Detected Colors', image)
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()  # Close the window
