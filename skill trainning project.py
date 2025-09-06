import cv2
import imutils
import numpy as np

# Initialize the HOG person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) on the luminance channel
def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final_image

# Apply gamma correction to improve brightness consistency
def adjust_gamma(image, gamma=1.2):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# Non-maxima suppression to filter out overlapping boxes
def non_max_suppression_fast(boxes, overlapThresh=0.65):
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")

# Open the video file or capture from a webcam (0 for webcam)
video = cv2.VideoCapture(r"C:\Users\jgmid\OneDrive\Pictures\Saved Pictures\project vdo.mp4")  # Change the path to your video file

# Skip every N frames to reduce processing load (optional)
frame_skip_interval = 2
frame_count = 0

# Loop over the video frames
while True:
    # Read the current frame
    ret, frame = video.read()

    # If the frame was not grabbed, then we've reached the end of the stream
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip_interval != 0:
        continue

    # Resize the frame for faster processing (keeping aspect ratio)
    frame = imutils.resize(frame, width=int(frame.shape[1] * 0.5))

    # Apply preprocessing techniques
    frame = apply_clahe(frame)  # Apply CLAHE for contrast enhancement
    frame = adjust_gamma(frame, gamma=1.2)  # Adjust gamma for brightness consistency

    # Detect pedestrians in the frame
    (regions, _) = hog.detectMultiScale(frame, 
                                        winStride=(8, 8),  # Increased stride for speed-up
                                        padding=(8, 8),    # Increased padding
                                        scale=1.05)        # Slightly smaller scale to reduce processing

    # Apply Non-Maxima Suppression to remove overlapping boxes
    regions = non_max_suppression_fast(regions, overlapThresh=0.65)

    # Draw rectangles around the detected pedestrians
    for (x, y, w, h) in regions:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the output frame
    cv2.imshow("Pedestrian Detection", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
video.release()
cv2.destroyAllWindows()