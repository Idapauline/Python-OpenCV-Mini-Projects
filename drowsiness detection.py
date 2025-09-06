import cv2
import numpy as np
import winsound # For alert sound on Windows

# Load the Haar Cascade classifiers for face and eyes
face_cascade = cv2.CascadeClassifier("C:\\Users\\jgmid\\.vscode\\haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("C:\\Users\\jgmid\\.vscode\\haarcascade_eye.xml")

# Define the Eye Aspect Ratio function
def eye_aspect_ratio(eye):
    if eye.shape[0] < 6:
        return 0  # Return 0 if not enough points

    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])

    ear = (A + B) / (2.0 * C)
    return ear

# Define the main function for drowsiness detection
def main():
    # Start video capture
    cap = cv2.VideoCapture(0)

    # Define parameters
    drowsiness_threshold = 0.25  # EAR threshold
    consecutive_frames = 0
    drowsiness_frames = 20  # Number of consecutive frames to trigger alert

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            # Detect eyes
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) >= 2:
                left_eye = eyes[0]
                right_eye = eyes[1]

                # Calculate EAR for both eyes
                left_eye_landmarks = np.array([(left_eye[0], left_eye[1]),
                                                (left_eye[0] + left_eye[2], left_eye[1]),
                                                (left_eye[0] + left_eye[2], left_eye[1] + left_eye[3]),
                                                (left_eye[0], left_eye[1] + left_eye[3]),
                                                (left_eye[0], left_eye[1])])
                
                right_eye_landmarks = np.array([(right_eye[0], right_eye[1]),
                                                 (right_eye[0] + right_eye[2], right_eye[1]),
                                                 (right_eye[0] + right_eye[2], right_eye[1] + right_eye[3]),
                                                 (right_eye[0], right_eye[1] + right_eye[3]),
                                                 (right_eye[0], right_eye[1])])

                left_eye_ratio = eye_aspect_ratio(left_eye_landmarks)
                right_eye_ratio = eye_aspect_ratio(right_eye_landmarks)

                if left_eye_ratio > 0 and right_eye_ratio > 0:
                    avg_ear = (left_eye_ratio + right_eye_ratio) / 2.0

                    # Check for drowsiness
                    if avg_ear < drowsiness_threshold:
                        consecutive_frames += 1
                        if consecutive_frames >= drowsiness_frames:
                            # Alert the driver
                            print("Drowsiness Detected!")
                            winsound.Beep(1000, 500)  # Sound alert
                    else:
                        consecutive_frames = 0  # Reset if eyes are open

        # Display the resulting frame
        cv2.imshow('Driver Drowsiness Detection', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
