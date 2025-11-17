import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('xray_twin1.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('xray_twin2.png', cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT_create()

keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

print("Number of keypoints in image 1:", len(keypoints1))
print("Number of keypoints in image 2:", len(keypoints2))

# match descriptors using brute-force KNN
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# apply Lowe Ratio Test to remove ambiguous matches
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print("Number of good matches:", len(good_matches))

# visualize matches
img_matches = cv2.drawMatches(
    img1, keypoints1, img2, keypoints2, good_matches, None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

plt.figure(figsize=(14, 8))
plt.imshow(img_matches, cmap='gray')
plt.title("Matches between two images (SIFT)")
plt.axis('off')
plt.show()
