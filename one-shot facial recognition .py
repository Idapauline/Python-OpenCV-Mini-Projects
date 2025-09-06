import cv2
import numpy as np

def extract_and_match_features(image1_path, image2_path):
    """
    Extracts and matches features between two images.

    Args:
        image1_path (str): Path to the first image.
        image2_path (str): Path to the second image.

    Returns:
        tuple: Tuple containing matched keypoints, keypoints from both images, and the images.
    """

    # Load the images in grayscale
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # Create SIFT detector
    sift = cv2.SIFT_create()

    # Detect and compute keypoints and descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # Create FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Match features
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    return good_matches, keypoints1, keypoints2, img1, img2


if __name__ == "__main__":

    image1_path = "C:\\Users\\jgmid\\OneDrive\\Pictures\\Saved Pictures\\image 1.jpg"
    image2_path = "C:\\Users\\jgmid\\OneDrive\\Pictures\\Saved Pictures\\image2.jpg"
    # Extract matched features and keypoints
    good_matches, keypoints1, keypoints2, img1, img2 = extract_and_match_features(image1_path, image2_path)

    # Draw the matched keypoints
    img3 = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=2)

    # Display the result
    cv2.imshow("Matched Keypoints", img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()