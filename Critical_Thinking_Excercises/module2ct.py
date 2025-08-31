import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread("puppy.jpg")
# Splitting using numpy array slicing because it is faster according to OpenCV documentation
# https://docs.opencv.org/4.x/d3/df2/tutorial_py_basic_ops.html
b_channel = img[:, :, 0]
g_channel = img[:, :, 1]
r_channel = img[:, :, 2]

# Create zero matrices for merging and showing individual channels
b_zero = np.zeros_like(b_channel)
g_zero = np.zeros_like(g_channel)
r_zero = np.zeros_like(r_channel)

# display individual channels in color
blue_img = cv2.merge((b_channel, b_zero, b_zero))
green_img = cv2.merge((b_zero, g_channel, g_zero))
red_img = cv2.merge((r_zero, r_zero, r_channel))
# Display individual channels
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(blue_img, cv2.COLOR_BGR2RGB))
plt.title("Blue Channel")
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(green_img, cv2.COLOR_BGR2RGB))
plt.title("Green Channel")
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(red_img, cv2.COLOR_BGR2RGB))
plt.title("Red Channel")
plt.axis('off')
plt.show()
# Merge channels back to form the original image
merged_img = cv2.merge((b_channel, g_channel, r_channel))
# Merge channels swapping R and G (G, R, B)
swapped_img = cv2.merge((b_channel, r_channel, g_channel))
# Display all images side by side in matplotlib
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(merged_img, cv2.COLOR_BGR2RGB))
plt.title("Merged Image")
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(swapped_img, cv2.COLOR_BGR2RGB))
plt.title("Swapped R and G")
plt.axis('off')
plt.show()
