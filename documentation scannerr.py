import cv2
import numpy as np

# Function to order the points of the document correctly (top-left, top-right, bottom-right, bottom-left)
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

# Function to apply the perspective transformation to get the top-down view
def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Compute the width of the new image (top and bottom width)
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # Compute the height of the new image (left and right height)
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Create destination points for the transformed image
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    # Get the transformation matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped

# Load the image
image = cv2.imread("C:\\Users\\jgmid\\OneDrive\\Pictures\\Saved Pictures\\document photo.jpg")

# Resize the image if it's too large for easier processing
height, width = image.shape[:2]
if height > 1000 or width > 1000:
    image = cv2.resize(image, (1000, int(1000 * height / width)))

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Edge detection using Canny (try adjusting the thresholds)
edges = cv2.Canny(blurred, 50, 150)

# You can try adaptive thresholding if Canny doesn't work well
# edges = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Find contours in the edged image
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

screenCnt = None

# Loop over the contours to find a rectangle (document)
for contour in contours:
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    
    # If the approximated contour has four points, then we assume we have found the document
    if len(approx) == 4:
        screenCnt = approx
        break

# Check if a valid contour was found
if screenCnt is None:
    print("No document-like contour was found.")
    # Show the edges image for debugging
    cv2.imshow("Edges", edges)
    cv2.waitKey(0)
else:
    # Apply the perspective transformation
    warped = four_point_transform(image, screenCnt.reshape(4, 2))

    # Convert the warped image to grayscale and then threshold it to give it the 'scanned' effect
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, scanned = cv2.threshold(warped_gray, 150, 255, cv2.THRESH_BINARY)

    # Display the result
    cv2.imshow("Original", image)
    cv2.imshow("Scanned", scanned)
    cv2.waitKey(0)
    cv2.destroyAllWindows()