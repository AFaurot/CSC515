import cv2
import numpy as np
import matplotlib.pyplot as plt

''' This is a script to perform the morphological operation of dilation, erosion, opening, and closing on a 
scanned image of handwriting in cursive. Adaptive thresholding is used to convert the grayscale image to binary.
This works better than global thresholding for this image.'''

# Read image in grayscale
img = cv2.imread("Module5CT.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(gray, 5)

# Adaptive thresholding had better results than global thresholding.
# Inspired by https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
binary_img = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 13, 4)

# establish a kernel
kernel = np.ones((5, 5), np.uint8)

# Morphological operations
dilation = cv2.dilate(binary_img, kernel, iterations=1)
erosion = cv2.erode(binary_img, kernel, iterations=1)
opening = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

# Put results in a dictionary for plotting
results = {
    "Original": cv2.cvtColor(img, cv2.COLOR_BGR2RGB),  # convert BGR→RGB for matplotlib
    "Gray": gray,
    "Binary": binary_img,
    "Dilation": dilation,
    "Erosion": erosion,
    "Opening": opening,
    "Closing": closing
}

# Plot results in a grid
plt.figure(figsize=(12, 8))
for i, (title, image) in enumerate(results.items(), 1):
    plt.subplot(2, 4, i)  # 2 rows, 4 columns
    if len(image.shape) == 2:  # grayscale/binary
        plt.imshow(image, cmap="gray")
    else:  # RGB color
        plt.imshow(image)
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()
