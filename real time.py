import cv2
import imutils

# Initialize HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Open the webcam video stream
cap = cv2.VideoCapture(0)  # 0 means the default webcam

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Resize the frame for faster processing
    frame = imutils.resize(frame, width=min(800, frame.shape[1]))

    # Detect people in the frame
    (pedestrians, _) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

    # Draw bounding boxes around detected people
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Pedestrian Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()




  