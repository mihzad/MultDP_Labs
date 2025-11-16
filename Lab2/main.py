import cv2
import numpy as np
import matplotlib.pyplot as plt


# grayscale - xray one
img = cv2.cvtColor(cv2.imread('xray.jpg'), cv2.COLOR_BGR2GRAY)
#cv2.imread('xray.jpg', cv2.IMREAD_GRAYSCALE)

# color - satellite one
img_color = cv2.imread('satellite.jpg')

cv2.imshow("Original", img)
cv2.waitKey(0)

#====== CANNY MANUAL ======

# 1)noise reduction with Gaussian filter

blur = cv2.GaussianBlur(img, (5, 5), 1.4)
cv2.imshow("Gaussian Blurred", blur)
cv2.waitKey(0)

# 2)Gradient calculation (Sobel operator)

Gx = cv2.Sobel(blur, cv2.CV_64F, dx=1, dy=0, ksize=3)
Gy = cv2.Sobel(blur, cv2.CV_64F, dx=0, dy=1, ksize=3)

magnitude = np.hypot(Gx, Gy)
magnitude = magnitude / magnitude.max() * 255  # normalize to 0–255
theta = np.arctan2(Gy, Gx) #grad direction

cv2.imshow("Gradient Magnitude", magnitude.astype(np.uint8))
cv2.waitKey(0)


# 3)Non-Maximum Suppression

def non_max_suppression(mag, angle):
    M, N = mag.shape
    Z = np.zeros((M, N), dtype=np.float32)
    angle = angle * 180. / np.pi
    angle[angle < 0] += 180 #we just need info thats vertical edge, no concrete details about it.

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            q = 255
            r = 255

            # Angle 0
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = mag[i, j + 1]
                r = mag[i, j - 1]
            # Angle 45
            elif (22.5 <= angle[i, j] < 67.5):
                q = mag[i + 1, j - 1]
                r = mag[i - 1, j + 1]
            # Angle 90
            elif (67.5 <= angle[i, j] < 112.5):
                q = mag[i + 1, j]
                r = mag[i - 1, j]
            # Angle 135
            elif (112.5 <= angle[i, j] < 157.5):
                q = mag[i - 1, j - 1]
                r = mag[i + 1, j + 1]

            if (mag[i, j] >= q) and (mag[i, j] >= r): # => local peak
                Z[i, j] = mag[i, j]
            else:
                Z[i, j] = 0
    return Z

nms = non_max_suppression(magnitude, theta)
cv2.imshow("After Non-Max Suppression", nms.astype(np.uint8))
cv2.waitKey(0)

# 4)Double Thresholding

highThreshold = 255 * 0.15 #nms is normalized to [0, 255]
lowThreshold = highThreshold * 0.5

print(f"high: {highThreshold}, low: {lowThreshold}")
res = np.zeros_like(nms)

#pixel values
strong = 255 
weak = 75

strong_i, strong_j = np.where(nms >= highThreshold)
weak_i, weak_j = np.where((nms <= highThreshold) & (nms >= lowThreshold))

res[strong_i, strong_j] = strong
res[weak_i, weak_j] = weak

cv2.imshow("Double Threshold", res.astype(np.uint8))
cv2.waitKey(0)

# 5)Edge Tracking by Hysteresis

from collections import deque

def hysteresis(img, weal_ptxval=75, strong_ptxval=255):
    """
    Perform hysteresis by connectivity: start from all strong pixels and
    flood-fill to connected weak pixels (8-connectivity).
    - img: 2D numpy array containing values {0, weal_ptxval, strong_ptxval}
           (float or int; will be processed in-place on a copy).
    - weal_ptxval: numeric marker for weak edges (default 75)
    - strong_ptxval: numeric marker for strong edges (default 255)
    Returns: binary edge map (dtype=np.uint8) with strong_ptxval as edges and 0 otherwise.
    """

    # Work on a copy to avoid modifying original outside
    img = img.copy().astype(np.uint8)

    # Find coordinates of strong pixels
    strong_yx = np.argwhere(img == strong_ptxval)
    if strong_yx.size == 0:
        # No strong pixels -> nothing to connect
        return np.zeros_like(img, dtype=np.uint8)

    # Use deque for BFS (or use list as stack for DFS)
    q = deque()
    for y, x in strong_yx:
        q.append((int(y), int(x)))

    # 8-neighborhood offsets
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 ( 0, -1),          ( 0, 1),
                 ( 1, -1), ( 1, 0), ( 1, 1)]

    H, W = img.shape

    while q:
        y, x = q.popleft()
        # Check all 8 neighbors
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= H or nx < 0 or nx >= W:
                continue
            if img[ny, nx] == weal_ptxval:
                # Promote weak to strong and add to queue
                img[ny, nx] = strong_ptxval
                q.append((ny, nx))

    # Any remaining weak|no-edge pixels are set to 0
    img[img != strong_ptxval] = 0
    return img.astype(np.uint8)


edges_manual = hysteresis(res.copy(), weal_ptxval=weak, strong_ptxval=strong)
cv2.imshow("Final Edges (manual Canny)", edges_manual.astype(np.uint8))
cv2.waitKey(0)

#====== CV2 CANNY ======

edges_cv = cv2.Canny(img, lowThreshold, highThreshold)
cv2.imshow("OpenCV Canny", edges_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()


