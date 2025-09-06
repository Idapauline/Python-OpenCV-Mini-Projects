import cv2
import numpy as np

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to predict mask based on color analysis
def predict_mask(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) for the face
        roi = frame[y:y+h, x:x+w]
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Define HSV range for detecting masks (this can vary based on actual mask colors)
        lower_mask_color = np.array([0, 48, 80])  # Lower HSV for mask color
        upper_mask_color = np.array([20, 255, 255])  # Upper HSV for mask color

        # Create a mask based on the defined HSV range
        mask = cv2.inRange(roi_hsv, lower_mask_color, upper_mask_color)
        mask_ratio = cv2.countNonZero(mask) / (w * h)

        # Threshold for mask detection based on mask coverage
        if mask_ratio > 0.2:  # Adjust this threshold as necessary
            label = 'Mask'
            color = (0, 255, 0)  # Green for mask
        else:
            label = 'No Mask'
            color = (0, 0, 255)  # Red for no mask

        # Draw rectangle around the face and put the label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return frame

# Load the image
image_path = "C:\\Users\\jgmid\\OneDrive\\Pictures\\Saved Pictures\\ms-dhoni-mass.jpg" # Replace with your image path
image = cv2.imread(image_path)

# Predict masks in the image
output_image = predict_mask(image)

# Display the output image
cv2.imshow('Mask Detection', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()