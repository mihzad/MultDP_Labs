import cv2
import numpy as np
from numpy.linalg import norm

def compute_hog(img):
    img = cv2.resize(img, (256, 256))  # consistent size

    hog = cv2.HOGDescriptor()
    features = hog.compute(img).flatten()
    return features


def hog_similarity(img1, img2):
    f1 = compute_hog(img1)
    f2 = compute_hog(img2)

    # cosine similarity
    cosine_sim = np.dot(f1, f2) / (norm(f1) * norm(f2))

    # euclidean distance
    euclid = norm(f1 - f2)

    return cosine_sim, euclid


# Example usage
img1 = cv2.imread('xray_twin1.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('xray_twin2.png', cv2.IMREAD_GRAYSCALE)

cos_sim, euc_dist = hog_similarity(img1, img2)

print("Cosine similarity:", cos_sim)
print("Euclidean distance:", euc_dist)
