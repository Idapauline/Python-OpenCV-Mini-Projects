import cv2
import numpy as np

def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces

def swap_faces(image1, image2):
    # Detect faces
    faces1 = detect_face(image1)
    faces2 = detect_face(image2)

    if len(faces1) == 0 or len(faces2) == 0:
        print("No faces found in one of the images.")
        return None

    # Assuming one face per image
    (x1, y1, w1, h1) = faces1[0]
    (x2, y2, w2, h2) = faces2[0]

    # Extract the faces
    face1 = image1[y1:y1 + h1, x1:x1 + w1]
    face2 = image2[y2:y2 + h2, x2:x2 + w2]

    # Resize faces to match the target
    face1_resized = cv2.resize(face1, (w2, h2))
    face2_resized = cv2.resize(face2, (w1, h1))

    # Swap faces
    image1[y1:y1 + h1, x1:x1 + w1] = face2_resized
    image2[y2:y2 + h2, x2:x2 + w2] = face1_resized

    return image1, image2

# Load images
image1 = cv2.imread('C:\Users\jgmid\OneDrive\Pictures\Saved Pictures\dhoni.jpg')
image2 = cv2.imread('C:\Users\jgmid\OneDrive\Pictures\Saved Pictures\ms-dhoni-mass.jpg')

# Perform face swapping
swapped_image1, swapped_image2 = swap_faces(image1, image2)

# Display the results
cv2.imshow("Swapped Image 1", swapped_image1)
cv2.imshow("Swapped Image 2", swapped_image2)
cv2.waitKey(0)
cv2.destroyAllWindows()