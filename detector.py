import argparse
import csv
import glob
import os

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
WORK_WIDTH = 300   # every photo is resized to this width before processing
IMAGE_TYPES = ("*.jpg", "*.jpeg", "*.png")


def load_image(path):
    """Read an image and resize it to WORK_WIDTH pixels wide."""
    image = cv2.imread(path)
    if image is None:
        return None

    # Resize so every photo is processed at the same scale. Without this, the
    # 5x5 cleaning kernel would be "big" on a small photo but "tiny" on a
    # 4000-pixel phone photo, and results would depend on camera resolution.
    scale = WORK_WIDTH / image.shape[1]
    return cv2.resize(image, (WORK_WIDTH, int(image.shape[0] * scale)))


def find_leaf_and_disease(image):
    """Return (leaf_mask, disease_mask) for a BGR image."""
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
    # Disease = anything inside the leaf that is NOT healthy green and NOT
    # glare. This catches brown, black, grey and yellow damage.
    healthy_mask = cv2.bitwise_or(green_mask, glare_mask)
    disease_mask = cv2.bitwise_and(leaf_mask, cv2.bitwise_not(healthy_mask))

    # "Opening" = erode then dilate. It deletes white areas thinner than the
    # kernel (the thin line around the leaf edge, tiny specks) but keeps
    # real disease patches at their original size.
    kernel = np.ones((5, 5), np.uint8)
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_OPEN, kernel)

    return leaf_mask, disease_mask


def categorise(percent):
    """Step 5: turn a disease percentage into a category."""
    if percent >= 10:
        return "Reject"
    elif percent >= 5:
        return "Eating purposes"
    else:
        return "Decorations"


def save_result(path, image, leaf_mask, disease_mask, percent, category):
    """Save a 4-panel figure to RESULTS_DIR and return the figure."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Paint the diseased pixels red on a copy of the original image
    highlighted = rgb.copy()
    highlighted[disease_mask > 0] = (255, 0, 0)

    titles = ["Original", "Leaf mask", "Disease mask",
              f"Disease: {percent:.2f}%"]
    images = [rgb, leaf_mask, disease_mask, highlighted]

    fig = plt.figure(figsize=(12, 4))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.title(titles[i])
        plt.axis("off")

    plt.suptitle(f"{os.path.basename(path)}  ->  {category}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    name = os.path.splitext(os.path.basename(path))[0]
    plt.savefig(os.path.join(RESULTS_DIR, f"{name}_result.png"), dpi=150)
    return fig


def analyse(path):
    """Run the whole pipeline on one photo. Returns a result dict or None."""
    image = load_image(path)
    if image is None:
        print(f"Skipping {path}: could not read image")
        return None

    leaf_mask, disease_mask = find_leaf_and_disease(image)

    # --- Step 4: calculate the disease percentage ---
    leaf_pixels = cv2.countNonZero(leaf_mask)
    disease_pixels = cv2.countNonZero(disease_mask)
    percent = disease_pixels / leaf_pixels * 100
    category = categorise(percent)

    fig = save_result(path, image, leaf_mask, disease_mask, percent, category)
    return {"image": os.path.basename(path),
            "disease_percent": round(percent, 2),
            "category": category,
            "figure": fig}


def collect_images(target):
    """A single file -> [file]; a folder -> every image inside it."""
    if os.path.isfile(target):
        return [target]
    paths = []
    for pattern in IMAGE_TYPES:
        paths += glob.glob(os.path.join(target, pattern))
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(
        description="Detect disease on betel leaves and categorise them.")
    parser.add_argument("target", nargs="?", default="images",
                        help="an image file or a folder of images "
                             "(default: images)")
    parser.add_argument("--show", action="store_true",
                        help="open a window for each result")
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")   # draw figures without opening windows

    paths = collect_images(args.target)
    if not paths:
        raise SystemExit(f"No images found in {args.target}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = []
    for path in paths:
        result = analyse(path)
        if result is None:
            continue
        results.append(result)
        if not args.show:
            plt.close(result["figure"])   # free memory in batch mode

    # --- Summary table ---
    print(f"\n{'Image':<20}{'Disease %':>10}   Category")
    print("-" * 48)
    for r in results:
        print(f"{r['image']:<20}{r['disease_percent']:>9.2f}%   {r['category']}")

    # --- Save the summary as a CSV file (opens in Excel) ---
    csv_path = os.path.join(RESULTS_DIR, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image", "disease_percent", "category"],
            extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved {len(results)} result images and {csv_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
