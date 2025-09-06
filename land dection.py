import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def draw_lines(img, lines):
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 5)

def lane_detection(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame
        frame = cv2.resize(frame, (640, 480))

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Define vertices for region of interest
        height, width = edges.shape
        roi_vertices = np.array([[(0, height), (width / 2, height / 2), (width, height)]], dtype=np.int32)
        roi = region_of_interest(edges, roi_vertices)

        # Hough transform to find lines
        lines = cv2.HoughLinesP(roi, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)

        # Draw lanes
        draw_lines(frame, lines)

        # Show the frame
        cv2.imshow("Lane Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Replace 'video_path.mp4' with the path to your video file
lane_detection("C:\\Users\\jgmid\\OneDrive\\Pictures\\Saved Pictures\\WhatsApp Video .mp4")
