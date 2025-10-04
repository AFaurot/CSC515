"""
Face detection + eye obfuscation script with face alignment for better eye detection.

Features:
- Loads all images from "images/" folder.
- Shows intermediate results (cv2.imshow); waits for user to close each window.
- Detects frontal faces with Haar cascade.
- Resizes large images to max dimension 800px while keeping aspect ratio.
- Preprocesses images with CLAHE for better contrast.
- After alignment, detects eyes and applies strong Gaussian blur only inside each eye bounding box.
- Pastes processed face back into original image.
- Skips blurring for faces where eyes are not detected.
- Saves results to "output/" folder.

Requirements: Python 3.x, OpenCV (cv2), numpy

Install with: pip install opencv-python numpy

How to use:
1. Create an "images/" folder in the same directory as this script.
2. Add images to the "images/" folder.
3. Run the script. It will process each image, show intermediate results, and save final images to "output/" folder.
"""
import cv2
import os
import numpy as np

# Configuration and global parameters
IMAGE_FOLDER = "images"
OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Pretrained Haar cascade paths for face and eye detection
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Face detection parameters
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 20
MIN_SIZE = (30, 30)

# Distant face detection parameters
DIST_SCALE_FACTOR = 1.06
DIST_MIN_NEIGHBORS = 9
DIST_MIN_SIZE = (15, 15)

# Strong Gaussian blur kernel size for eye obfuscation
GAUSSIAN_KSIZE = (51, 51)  # strong blur


# blur only inside detected eye regions
def strong_blur_region(img, x1, y1, x2, y2):
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return img
    roi = img[y1:y2, x1:x2]
    blurred = cv2.GaussianBlur(roi, GAUSSIAN_KSIZE, 0)
    img[y1:y2, x1:x2] = blurred
    return img


# rotate face so eyes are horizontal, return rotated face and angle for reverse rotation
def align_face(face_img, eye_coords):
    # Rotate only if two eyes detected
    if len(eye_coords) < 2:
        return face_img, 0
    (x1, y1, w1, h1), (x2, y2, w2, h2) = eye_coords[:2]
    eye_center1 = (x1 + w1//2, y1 + h1//2)
    eye_center2 = (x2 + w2//2, y2 + h2//2)
    # Calculate difference in y and x between the two eye centers
    dy = eye_center2[1] - eye_center1[1]
    dx = eye_center2[0] - eye_center1[0]
    # Calculate angle in degrees using the arc tangent
    angle = np.degrees(np.arctan2(dy, dx))
    # Cast coordinates to Python int to avoid OpenCV TypeError
    eyes_center = (int((eye_center1[0] + eye_center2[0]) // 2),
                   int((eye_center1[1] + eye_center2[1]) // 2))
    rot_mat = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
    # Border replicate to avoid black borders after rotation
    aligned_face = cv2.warpAffine(
        face_img, rot_mat, (face_img.shape[1], face_img.shape[0]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    return aligned_face, angle


# Rotate face back to original orientation
def rotate_back(face_img, angle):
    if angle == 0:
        return face_img
    center = (face_img.shape[1]//2, face_img.shape[0]//2)
    rot_mat = cv2.getRotationMatrix2D(center, -angle, 1.0)
    restored_face = cv2.warpAffine(
        face_img, rot_mat, (face_img.shape[1], face_img.shape[0]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    return restored_face


# Primary function to process each image
def process_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load {img_path}")
        return

    cv2.imshow("Original", img)
    cv2.waitKey(0)

    # Resize if too large and keep aspect ratio
    max_dim = 800
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Add CLAHE to enhance contrast adaptively
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Show grayscale image
    cv2.imshow("Grayscale", gray)
    cv2.waitKey(0)
    # Apply global parameters depending on face distance (indicated by filename)
    if "distant" in img_path.lower():
        scaleFactor = DIST_SCALE_FACTOR
        minNeighbors = DIST_MIN_NEIGHBORS
        minSize = DIST_MIN_SIZE
    else:
        scaleFactor = SCALE_FACTOR
        minNeighbors = MIN_NEIGHBORS
        minSize = MIN_SIZE

    # Detect faces
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize
    )

    # Draw face boxes
    vis_faces = img.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(vis_faces, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.imshow("Face detections (red)", vis_faces)
    cv2.waitKey(0)

    final = img.copy()
    # Process only if faces detected
    if len(faces) == 0:
        print(f"No faces detected in {os.path.basename(img_path)}; skipping obfuscation.")
    else:
        # Process each detected face
        for (fx, fy, fw, fh) in faces:
            # Face_color is for color image, face_gray is used for detection tasks
            face_color = final[fy:fy+fh, fx:fx+fw].copy()
            face_gray = gray[fy:fy+fh, fx:fx+fw]

            # Upscale distant faces for better eye detection
            if "distant" in img_path.lower():
                # Upscale face region by 2x
                face_color = cv2.resize(face_color, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
                face_gray = cv2.resize(face_gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

            # Initial eye detection for horizontal alignment
            eyes_initial = EYE_CASCADE.detectMultiScale(face_gray, scaleFactor=1.02, minNeighbors=20, minSize=(10,10))
            aligned_face, angle = align_face(face_color, eyes_initial)

            # Detect eyes again on aligned face
            aligned_gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
            eyes_final = EYE_CASCADE.detectMultiScale(aligned_gray, scaleFactor=1.02, minNeighbors=20, minSize=(10,10))

            # Blur detected eyes only
            for (ex, ey, ew, eh) in eyes_final:
                ex1 = max(ex - ew//8, 0)
                ey1 = max(ey - eh//8, 0)
                ex2 = min(ex + ew + ew//8, aligned_face.shape[1])
                ey2 = min(ey + eh + eh//8, aligned_face.shape[0])
                # Draw box for visualization - Uncomment line below for testing where the boundary box is
                # cv2.rectangle(aligned_face, (ex1, ey1), (ex2, ey2), (0, 255, 255), 2)
                aligned_face = strong_blur_region(aligned_face, ex1, ey1, ex2, ey2)

            # Rotate face back to original orientation
            aligned_face = rotate_back(aligned_face, angle)

            if "distant" in img_path.lower():
                # Downscale back to original face size
                aligned_face = cv2.resize(aligned_face, (fw, fh), interpolation=cv2.INTER_LINEAR)

            # Paste processed face back
            final[fy:fy+fh, fx:fx+fw] = aligned_face

    # Show final result
    cv2.imshow("Final - Eyes Obfuscated", final)
    cv2.waitKey(0)
    # Save output
    out_name = os.path.basename(img_path)
    out_path = os.path.join(OUTPUT_FOLDER, f"blurred_{out_name}")
    cv2.imwrite(out_path, final)
    print(f"Saved: {out_path}")


# Main processing loop
def main():
    if not os.path.exists(IMAGE_FOLDER):
        print(f"Image folder '{IMAGE_FOLDER}' does not exist. Create it and add images.")
        return

    files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        print(f"No images found in '{IMAGE_FOLDER}'.")
        return

    for f in files:
        print(f"Processing {f} ...")
        process_image(os.path.join(IMAGE_FOLDER, f))

    cv2.destroyAllWindows()
    print("All done. Outputs are in the 'output' folder.")


if __name__ == "__main__":
    main()
