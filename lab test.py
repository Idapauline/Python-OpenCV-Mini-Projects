import cv2
import numpy as np

# Function to detect specific colors and display their names
def detect_color(image_path, hsv_ranges):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert the image to HSV (Hue, Saturation, Value) color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    detected_colors = []

    # Iterate through the provided color ranges
    for color_name, (lower_color, upper_color) in hsv_ranges.items():
        # Create a mask for the specified color range
        mask = cv2.inRange(hsv_image, lower_color, upper_color)
        
        # Check if any pixels in the mask are detected
        if cv2.countNonZero(mask) > 0:
            detected_colors.append(color_name)
            
            # Apply the mask to the original image
            result = cv2.bitwise_and(image, image, mask=mask)
            cv2.imshow(f'{color_name} Detected', result)

    # Show the original image
    cv2.imshow('Original Image', image)
    
    # Display the names of the detected colors
    if detected_colors:
        print(f"Detected colors: {', '.join(detected_colors)}")
    else:
        print("No specified colors detected.")
    
    # Wait until a key is pressed, then close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":

    # Define color ranges (HSV format) for multiple colors
    hsv_ranges = {
        'Red': (np.array([0, 120, 70]), np.array([10, 255, 255])),     # Red range
        'Green': (np.array([36, 100, 100]), np.array([86, 255, 255])),  # Green range
        'Blue': (np.array([94, 80, 2]), np.array([126, 255, 255]))      # Blue range
    }
    
    # Path to the image
    image_path ="C:\\Users\\jgmid\\OneDrive\\Pictures\\Saved Pictures\\day2image.png"  # Replace with your image path
    
    # Call the color detection function
    detect_color(image_path, hsv_ranges)