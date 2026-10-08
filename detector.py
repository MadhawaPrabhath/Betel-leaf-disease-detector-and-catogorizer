import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_PATH = "images/leaf1.jpg"
RESULTS_DIR = "results"
WORK_WIDTH = 300   # every photo is resized to this width before processing

image = cv2.imread(IMAGE_PATH)
if image is None:
    raise SystemExit("Could not load image - check the file name/path")

# Resize so every photo is processed at the same scale. Without this, the
# 5x5 cleaning kernel would be "big" on a small photo but "tiny" on a
# 4000-pixel phone photo, and results would depend on camera resolution.
scale = WORK_WIDTH / image.shape[1]
image = cv2.resize(image, (WORK_WIDTH, int(image.shape[0] * scale)))

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# --- Step 1: colour masks (white = pixel matches, black = doesn't) ---
#                          lower (H, S, V)   upper (H, S, V)
green_mask = cv2.inRange(hsv, (25, 40, 30), (90, 255, 255))
brown_mask = cv2.inRange(hsv, (0, 40, 20), (25, 255, 170))

# Shiny reflections on the leaf: either washed-out (low saturation, bright)
# or bluish-white (hue above green, bright). These are healthy, not disease.
glare_mask = cv2.bitwise_or(
    cv2.inRange(hsv, (0, 0, 140), (179, 60, 255)),
    cv2.inRange(hsv, (90, 0, 110), (179, 255, 255)))

# The leaf is everything that is green OR dark brown
leaf_mask = cv2.bitwise_or(green_mask, brown_mask)

# --- Step 2: clean the mask ---
# Find the outlines of all white regions, keep the biggest one (the leaf),
# and fill it in solid. This removes small specks of noise and fills
# shiny spots inside the leaf that didn't match any colour.
contours, _ = cv2.findContours(
    leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest = max(contours, key=cv2.contourArea)

leaf_mask = np.zeros_like(leaf_mask)
cv2.drawContours(leaf_mask, [largest], -1, 255, cv2.FILLED)

# --- Step 3: find the diseased area ---
# Disease = anything inside the leaf that is NOT healthy green and NOT glare.
# This catches brown, black, grey and yellow damage, not just brown.
healthy_mask = cv2.bitwise_or(green_mask, glare_mask)
disease_mask = cv2.bitwise_and(leaf_mask, cv2.bitwise_not(healthy_mask))

# "Opening" = erode then dilate. It deletes white areas thinner than the
# kernel (the thin line around the leaf edge, tiny specks) but keeps
# real disease patches at their original size.
kernel = np.ones((5, 5), np.uint8)
disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_OPEN, kernel)

# --- Step 4: calculate the disease percentage ---
leaf_pixels = cv2.countNonZero(leaf_mask)
disease_pixels = cv2.countNonZero(disease_mask)
disease_percent = disease_pixels / leaf_pixels * 100


# --- Step 5: categorise the leaf ---
def categorise(percent):
    if percent >= 10:
        return "Reject"
    elif percent >= 5:
        return "Eating purposes"
    else:
        return "Decorations"


category = categorise(disease_percent)

print("Image:                ", IMAGE_PATH)
print("Leaf area (pixels):   ", leaf_pixels)
print("Disease area (pixels):", disease_pixels)
print(f"Disease percentage:    {disease_percent:.2f}%")
print("Category:             ", category)

# Paint the diseased pixels red on a copy of the original image
highlighted = rgb.copy()
highlighted[disease_mask > 0] = (255, 0, 0)

# --- Show and save the results ---
titles = ["Original", "Leaf mask", "Disease mask",
          f"Disease: {disease_percent:.2f}%"]
images = [rgb, leaf_mask, disease_mask, highlighted]

plt.figure(figsize=(12, 4))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.suptitle(f"{os.path.basename(IMAGE_PATH)}  ->  {category}",
             fontsize=14, fontweight="bold")
plt.tight_layout()

# Save BEFORE plt.show(): once the window is closed the figure is gone
os.makedirs(RESULTS_DIR, exist_ok=True)
name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
save_path = os.path.join(RESULTS_DIR, f"{name}_result.png")
plt.savefig(save_path, dpi=150)
print("Saved result to:      ", save_path)

plt.show()
