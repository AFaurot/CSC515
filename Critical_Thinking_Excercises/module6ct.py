import cv2
import matplotlib.pyplot as plt


def adaptive_threshold(image_path, blur_type='median', blur_ksize=5, blk_size=11, C=2):
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Apply blur to reduce small noise before thresholding
    if blur_type == 'gaussian':
        blurred = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
    else:  # default to median blur
        blurred = cv2.medianBlur(img, blur_ksize)

    # Adaptive thresholding (Gaussian method)
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blk_size,                      # block size (area for threshold calculation)
        C                        # constant subtracted from mean
    )

    # Display results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(blurred, cmap='gray')
    plt.title("Blurred")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(adaptive_thresh, cmap='gray')
    plt.title("Adaptive Threshold")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Optionally save result to file with descriptive name
    print(" Would you like to save the result? (y/n)")
    save_choice = input().strip().lower()
    if save_choice == 'y':
        print(f"Saving the result as {image_path.split('.')[0]}_{blur_type}{blur_ksize}_BLK{blk_size}C{C}.jpg")
        save_path = f"{image_path.split('.')[0]}_{blur_type}{blur_ksize}_BLK{blk_size}C{C}.jpg"
        cv2.imwrite(save_path, adaptive_thresh)
    else:
        print("Result not saved.")


def main():

    # Execute in a loop to allow multiple tests
    while True:
        print("Which image would you like to process?")
        print("1 for indoor.jpg, 2 for outdoor.jpg, 3 for closeup.jpg")
        img_choice = int(input("Enter choice (1-3) or 0 to quit: "))
        image_file= None
        if img_choice == 0:
            break
        if img_choice == 1:
            image_file = "indoor.jpg"
        elif img_choice == 2:
            image_file = "outdoor.jpg"
        elif img_choice == 3:
            image_file = "closeup.jpg"
        else:
            print("Invalid choice, please try again.")
        print(f"Which blur method would you like to use for {image_file}?")
        blur_choice = int(input("1 for median blur, 2 for gaussian blur: "))
        if blur_choice == 1:
            blur_type = 'median'
        else:
            blur_type = 'gaussian'
        print("Which blur kernel size would you like to use?")
        blur_ksize = int(input("Enter odd integer (3, 5, 7, ...): "))
        if blur_ksize % 2 == 0:
            print("Kernel size must be an odd integer. Using default size 5.")
            blur_ksize = 5
        print("Enter adaptive threshold block size (odd integer >=3, default is 11):")
        blk_size = int(input())
        if blk_size < 3 or blk_size % 2 == 0:
            print("Block size must be an odd integer >=3. Using default size 11.")
            blk_size = 11
        print("Enter adaptive threshold C value (default is 2):")
        C = int(input())
        adaptive_threshold(image_file, blur_type=blur_type, blur_ksize=blur_ksize, blk_size=blk_size, C=C)


if __name__ == "__main__":
    main()
