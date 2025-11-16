import cv2
import numpy as np


img = cv2.cvtColor(cv2.imread('xray_peppered.png'), cv2.COLOR_BGR2GRAY)
#cv2.imread('xray.jpg', cv2.IMREAD_GRAYSCALE)

cv2.imshow("Original", img)
cv2.waitKey(0)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

_, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

erosion = cv2.erode(binary, kernel, iterations=2)
cv2.imshow("Erosion", erosion)

dilation = cv2.dilate(binary, kernel, iterations=1)
cv2.imshow("Dilation", dilation)

opening = cv2.dilate(erosion, kernel, iterations=1)
closing = cv2.erode(dilation, kernel, iterations=1)
cv2.imshow("Opening", opening)
cv2.imshow("Closing", closing)
cv2.waitKey(0)
cv2.destroyAllWindows()