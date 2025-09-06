import cv2

# Create a background subtractor object
backSub = cv2.createBackgroundSubtractorMOG2()

# Load the video
video_path = "C:\\Users\\jgmid\\Downloads\\2103099-uhd_3840_2160_30fps.mp4"  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

while True:
    # Read a new frame
    ret, frame = cap.read()
    
    if not ret:
        break

    # Apply the background subtractor to get the foreground mask
    fg_mask = backSub.apply(frame)

    # Optional: Use morphological operations to reduce noise in the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours of the moving objects
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around detected moving objects
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter out small movements
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Motion Detection', frame)

    # Press 'q' to quit the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()