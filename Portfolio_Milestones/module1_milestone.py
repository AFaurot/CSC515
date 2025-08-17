import cv2

# Load an image from file
image = cv2.imread("shutterstock93075775--250.jpg")
# Display the image
cv2.imshow("Brain Image", image)
# Wait for a key press
cv2.waitKey(0)
# Write the image to a file
cv2.imwrite("C:\\Users\\freer\\PycharmProjects\\CSC515\\brain_image_copy.jpg", image)
# Close all OpenCV windows
cv2.destroyAllWindows()