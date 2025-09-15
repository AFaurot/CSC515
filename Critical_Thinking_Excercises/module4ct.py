import cv2
import matplotlib.pyplot as plt

# Kernel sizes to test
KERNEL_SIZES = [3, 5, 7]
# Two sigma values for Gaussian blur
SIGMA_VALUES = [0, 1.0]

# Load image
img = cv2.imread("Mod4CT1.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Store results
means = []
medians = []
gaussians_sigma0 = []
gaussians_sigma2 = []

for k in KERNEL_SIZES:
    # Mean
    means.append(cv2.blur(img, (k, k)))
    # Median
    medians.append(cv2.medianBlur(img, k))
    # Gaussian with sigma=0 (auto)
    gaussians_sigma0.append(cv2.GaussianBlur(img, (k, k), SIGMA_VALUES[0]))
    # Gaussian with sigma=2.0
    gaussians_sigma2.append(cv2.GaussianBlur(img, (k, k), SIGMA_VALUES[1]))

# Plot results: 3 rows (kernel sizes), 4 columns (Mean, Median, Gaussian sigma=0, Gaussian sigma=2)
fig, axes = plt.subplots(len(KERNEL_SIZES), 4, figsize=(20, 6 * len(KERNEL_SIZES)))

col_titles = ["Mean", "Median", "Gaussian (sigma=0)", "Gaussian (sigma=1.0)"]

# Column headers
for ax, title in zip(axes[0], col_titles):
    ax.set_title(title, fontsize=14)

# Fill rows
for i, k in enumerate(KERNEL_SIZES):
    axes[i, 0].imshow(means[i])
    axes[i, 0].axis("off")
    axes[i, 0].set_ylabel(f"{k}x{k}", fontsize=14, rotation=90)

    axes[i, 1].imshow(medians[i])
    axes[i, 1].axis("off")

    axes[i, 2].imshow(gaussians_sigma0[i])
    axes[i, 2].axis("off")

    axes[i, 3].imshow(gaussians_sigma2[i])
    axes[i, 3].axis("off")

plt.tight_layout()
plt.show()
