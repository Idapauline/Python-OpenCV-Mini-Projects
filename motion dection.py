import cv2

def motion_detection():
    # Capture video from the webcam (use a file path instead of 0 for a video file)
    cap = cv2.VideoCapture(0)

    # Initialize the first frame to compare with subsequent frames
    ret, frame1 = cap.read()
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame1_gray = cv2.GaussianBlur(frame1_gray, (21, 21), 0)

    while cap.isOpened():
        # Capture the next frame
        ret, frame2 = cap.read()
        if not ret:
            break

        # Convert the current frame to grayscale and apply Gaussian blur
        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        frame2_gray = cv2.GaussianBlur(frame2_gray, (21, 21), 0)

        # Calculate the difference between the first frame and the current frame
        diff = cv2.absdiff(frame1_gray, frame2_gray)

        # Apply thresholding to highlight the areas where motion is detected
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Find contours of the moving objects
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False  # Variable to check if motion is detected

        # Draw rectangles around the detected motion
        for contour in contours:
            if cv2.contourArea(contour) > 1000:  # Adjust contour area threshold for sensitivity
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)
                motion_detected = True  # Set to True if motion is detected

        # If motion is detected, add the text "Motion Detected" to the video stream
        if motion_detected:
            cv2.putText(frame2, "Motion Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # Display the original video feed with rectangles and text
        cv2.imshow("Motion Detection", frame2)

        # Update the first frame to the current frame for comparison in the next iteration
        frame1_gray = frame2_gray

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    motion_detection()