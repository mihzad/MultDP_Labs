import cv2
import numpy as np
import matplotlib.pyplot as plt


# grayscale - xray one
img_gray = cv2.cvtColor(cv2.imread('xray.jpg'), cv2.COLOR_BGR2GRAY)
#cv2.imread('xray.jpg', cv2.IMREAD_GRAYSCALE)

# color - satellite one
img_color = cv2.imread('satellite.jpg')

#==============================
#========= GRAYSCALE ==========
#==============================

#====== Original ======

def plot_histogram(image, title="Histogram"):
    plt.figure()
    plt.title(title)
    plt.xlabel("Intensity")
    plt.ylabel("Count")
    plt.hist(image.ravel(), 256, (0, 256))
    plt.show()

plot_histogram(img_gray, "Original Grayscale Histogram")

#====== Equalization ======
equalized_img = cv2.equalizeHist(img_gray)
plot_histogram(equalized_img, "Equalized Grayscale Histogram")

cv2.imshow("Original Gray", img_gray)
cv2.imshow("Equalized Gray", equalized_img)
cv2.waitKey(0)


#====== CLAHE ======
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_img = clahe.apply(img_gray)

plot_histogram(clahe_img, "CLAHE Grayscale Histogram")

cv2.imshow("CLAHE Gray", clahe_img)
cv2.waitKey(0)


#====== sharpening ======

example_sharpen_kernel = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]])

sharpened_clahe = cv2.filter2D(clahe_img, -1, example_sharpen_kernel)
sharpened_equalized = cv2.filter2D(equalized_img, -1, example_sharpen_kernel)

cv2.imshow("Sharpened CLAHE Gray", sharpened_clahe)
cv2.imshow("Sharpened Equalized Gray", sharpened_equalized)
cv2.waitKey(0)


#==============================
#=========== COLOR ============
#==============================

# equalizeHist() and clahe.apply() are only capable of one-channel equalization,
# perturbing color gamma otherwise.
# The solution found is to use non-RGB formats that describe brightness as a separate channel.
# I found that both YCrCb (with its Y) and LAB with its (L) do the trick.

def equalize_color(img_bgr): #yrcrb example
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)
    Y_eq = cv2.equalizeHist(Y)
    merged = cv2.merge((Y_eq, Cr, Cb))
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

def clahe_color(img_bgr): #lab example
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    L_clahe = clahe.apply(L)
    merged = cv2.merge((L_clahe, A, B))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

eq_color_img = equalize_color(img_color)
clahe_color_img = clahe_color(img_color)

#====== sharpening ======
sharp_clahe_color = cv2.filter2D(clahe_color_img, -1, example_sharpen_kernel)
sharp_eq_color = cv2.filter2D(eq_color_img, -1, example_sharpen_kernel)

#====== results ======
cv2.imshow("Original Color", img_color)
cv2.imshow("Equalized Color", eq_color_img)
cv2.imshow("CLAHE Color", clahe_color_img)
cv2.imshow("Sharpened CLAHE Color", sharp_clahe_color)
cv2.imshow("Sharpened Equalized Color", sharp_eq_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
