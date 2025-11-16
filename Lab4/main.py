import cv2
import numpy as np


def otsu_threshold(image):
    pixels_flattened = image.ravel()
    hist, _ = np.histogram(pixels_flattened, bins=256, range=(0, 256))
    total = pixels_flattened.size
    class_probabilities = hist / total
    #cumulative prob[i] = probability of pixel to have intensity <= i
    cumulative_prob = np.cumsum(class_probabilities)
    #cumulative_mean[i] =  intensity expectatation between pixels <= i
    cumulative_mean = np.cumsum(class_probabilities * np.arange(256))
    global_mean = cumulative_mean[-1] #expectation between all the pixels = whole img

    #between-class var for all the thresholds at once
    sigma_b_squared = (global_mean * cumulative_prob - cumulative_mean)**2 / (
        cumulative_prob * (1 - cumulative_prob) + 1e-7
    )

    T = np.argmax(sigma_b_squared)
    binary = np.where(image > T, 255, 0).astype(np.uint8)
    return T, binary


img = cv2.cvtColor(cv2.imread('xray_peppered.png'), cv2.COLOR_BGR2GRAY)

cv2.imshow("Original", img)
cv2.waitKey(0)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))


_, binary_my = otsu_threshold(img)

_, binary_cv2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imshow("Otsu: my", binary_my)
cv2.imshow("Otsu: cv2", binary_cv2)

cv2.waitKey(0)
cv2.destroyAllWindows()