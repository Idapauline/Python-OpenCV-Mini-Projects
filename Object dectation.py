import cv2
import numpy as np

# Load the video or capture from the webcam
cap = cv2.VideoCapture(0)  # Replace with 0 to use webcam

# Take the first frame of the video
ret, frame = cap.read()

# Select the ROI (Region of Interest) manually
roi = cv2.selectROI(frame, False)
x, y, w, h = roi
track_window = (x, y, w, h)

# Crop the ROI from the frame
roi_frame = frame[y:y+h, x:x+w]

# Convert the ROI to HSV (Hue-Saturation-Value) color space
hsv_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

# Create a mask and calculate the histogram of the ROI in the HSV space
roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Setup the termination criteria: either 10 iterations or at least 1 pt movement
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Back-projection of the ROI histogram to the current frame
    back_proj = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # Use MeanShift for object tracking
    ret, track_window = cv2.meanShift(back_proj, track_window, term_crit)

    # Draw a rectangle around the tracked object using MeanShift
    x, y, w, h = track_window
    mean_shift_img = cv2.rectangle(frame.copy(), (x, y), (x+w, y+h), 255, 2)

    # Display the MeanShift result
    cv2.imshow('MeanShift Tracking', mean_shift_img)

    # Use CamShift for object tracking (adaptive window size and angle)
    ret, track_window = cv2.CamShift(back_proj, track_window, term_crit)

    # Draw a rotated rectangle around the tracked object using CamShift
    pts = cv2.boxPoints(ret)
    pts = np.int0(pts)
    cam_shift_img = cv2.polylines(frame.copy(), [pts], True, 255, 2)

    # Display the CamShift result
    cv2.imshow('CamShift Tracking', cam_shift_img)

    # Break loop on 'ESC' key
    key = cv2.waitKey(30) & 0xFF
    if key == 27:  # ESC key
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()