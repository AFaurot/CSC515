import cv2


# This program draws bounding boxes around my face and eyes in an image and labels the image with "This is me!" text.
# It is inspired by the OpenCV face detection example at
# https://www.geeksforgeeks.org/python/opencv-python-program-face-detection/
# This program does not use a webcam capture like in above example, but rather a static image file "me.jpg"
def main():

    # Load image of myself from file
    img = cv2.imread("me.jpg")
    # resize image to 647 x 728 (50 percent of original size)
    img = cv2.resize(img, (647, 728))
    # Load the pre-trained classifiers for face and eyes
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    # Convert the image to grayscale for detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    # Draw circular bounding boxes around detected faces in green and eyes in red rectangles
    for (x, y, w, h) in faces:
        center = (x + w // 2, y + h // 2)
        radius = w // 2
        cv2.circle(img, center, radius, (0, 255, 0), 2)  # Green circle for face
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)  # Red rectangle for eyes
    # Add "This is me!" text at the top center of the image
    cv2.putText(img, "This is me!", (250, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Show final image with bounding boxes and text
    cv2.imshow("Detected Face and Eyes", img)
    cv2.waitKey(0)
    # Save and display the result
    cv2.imwrite("me_detected.jpg", img)
    # Close all OpenCV windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
