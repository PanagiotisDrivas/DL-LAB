import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from config import *

# ==========================================================
# IO
# ==========================================================

PALETTE = {
    "road": "#804080",
    "sidewalk": "#F423E8",
    "building": "#464646",
    "wall": "#66669C",
    "fence": "#BE9999",
    "pole": "#999999",
    "traffic light": "#FAAA1E",
    "traffic sign": "#DCDC00",
    "vegetation": "#6B8E23",
    "terrain": "#98FB98",
    "sky": "#4682B4",
    "person": "#DC143C",
    "rider": "#FF0000",
    "car": "#00008E",
    "truck": "#000046",
    "bus": "#003C64",
    "train": "#005064",
    "motorcycle": "#0000E6",
    "bicycle": "#770B20",
}


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


PALETTE_RGB = {
    cls: np.array(hex_to_rgb(color), dtype=np.uint8)
    for cls, color in PALETTE.items()
}

def create_output_dirs():
    for cls in CLASSES:
        (OUTPUT_REFERENCE_SEMANTIC_BANK_ROOT / cls).mkdir(parents=True, exist_ok=True)


def get_json_files(split="train"):

    return sorted((CITYSCAPES_ROOT / "gtFine" / split).rglob("*_gtFine_polygons.json"))

def get_color_files(split="train"):
    return sorted(
        (CITYSCAPES_ROOT / "gtFine" / split).rglob("*_gtFine_color.png")
    )

def get_image_path(color_path, split="train"):

    city = color_path.parent.name

    stem = color_path.stem.replace("_gtFine_color", "")

    return (
        CITYSCAPES_ROOT
        / "leftImg8bit_sequence"
        / split
        / city
        / f"{stem}_leftImg8bit.png"
    )

def load_color_label(path):

    img = cv2.imread(str(path))

    if img is None:
        return None

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_annotation(json_path):

    with open(json_path, "r") as f:
        return json.load(f)


def load_image(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        return None

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ==========================================================
# Save reference
# ==========================================================


def save_reference(label, image, mask, counters):

    if counters[label] >= MAX_PER_CLASS:
        return

    idx = counters[label]

    image_file = OUTPUT_REFERENCE_SEMANTIC_BANK_ROOT / label / f"{label}_{idx:03d}_img.png"

    mask_file = OUTPUT_REFERENCE_SEMANTIC_BANK_ROOT / label / f"{label}_{idx:03d}_mask.png"

    cv2.imwrite(str(image_file), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    cv2.imwrite(str(mask_file), mask)

    counters[label] += 1


def process_annotation(color_file, counters, split="train"):

    image_path = get_image_path(color_file, split)

    if not image_path.exists():
        return

    image = load_image(image_path)

    if image is None:
        return

    label_img = load_color_label(color_file)

    if label_img is None:
        return

    # Number of references that must satisfy MIN_AREA
    STRICT_EXAMPLES = 10

    for label in CLASSES:

        # Skip if we've already collected enough references
        if counters[label] >= MAX_PER_CLASS:
            continue

        if label not in PALETTE_RGB:
            continue

        color = PALETTE_RGB[label]

        # Create binary mask
        mask = np.all(label_img == color, axis=-1).astype(np.uint8) * 255

        area = np.count_nonzero(mask)

        # For the first STRICT_EXAMPLES references, enforce MIN_AREA.
        # Afterwards, allow smaller instances.
        if counters[label] < STRICT_EXAMPLES:
            if area < MIN_AREA:
                continue
        else:
            # Skip empty masks
            if area == 0:
                continue

        save_reference(label, image, mask, counters)


# ==========================================================
# Main
# ==========================================================


def build_reference_bank(split="train", min_instances=3):

    create_output_dirs()

    counters = {cls: 0 for cls in CLASSES}

    # json_files = get_json_files(split)

    color_files = get_color_files(split)

    print(f"Found {len(color_files)} label images.")

    for idx, color_file in enumerate(color_files, start=1):

        process_annotation(color_file, counters, split)

        if idx % 100 == 0:
            print(f"{idx}/{len(color_files)} processed")

        if all(counters[c] >= MAX_PER_CLASS for c in CLASSES):
            break


    print("\nFinished\n")

    print("=" * 40)
    print("Reference bank summary")
    print("=" * 40)

    for cls in CLASSES:

        print(f"{cls:15s}: {counters[cls]}")


if __name__ == "__main__":

    build_reference_bank(split="train", min_instances=3)
