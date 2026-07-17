import json
from pathlib import Path

import cv2
import numpy as np

from config import *

# ==========================================================
# Configuration
# ==========================================================

# ==========================================================
# IO
# ==========================================================


def create_output_dirs():
    for cls in CLASSES:
        (OUTPUT_REFRENCE_INSTANCE_BANK_ROOT / cls).mkdir(parents=True, exist_ok=True)
        (OUTPUT_REFRENCE_INSTANCE_BANK_ROOT / cls).mkdir(parents=True, exist_ok=True)


def get_json_files(split="train"):
    return sorted((CITYSCAPES_ROOT / "gtFine" / split).rglob("*_gtFine_polygons.json"))


def get_image_path(json_path: Path, split="train"):

    city = json_path.parent.name

    stem = json_path.stem.replace("_gtFine_polygons", "")

    return (
        CITYSCAPES_ROOT
        / "leftImg8bit_sequence"
        / split
        / city
        / f"{stem}_leftImg8bit.png"
    )


def load_annotation(json_path):

    with open(json_path, "r") as f:
        return json.load(f)


def load_image(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        return None

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ==========================================================
# Object utilities
# ==========================================================


def is_valid_object(obj):
    """
    Validate object label and polygon.
    """

    label = obj["label"]

    # Ignore Cityscapes labels not in our classes
    if label not in CLASS_SET:
        return False, None

    polygon = np.asarray(obj["polygon"], dtype=np.int32)

    if len(polygon) < 3:
        return False, None

    return True, polygon


def create_mask(image_shape, polygon):
    """
    Create full resolution binary mask.

    White  = object
    Black  = background

    Returns:
        mask
        white pixel area
    """

    h, w = image_shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)

    cv2.fillPoly(mask, [polygon], 255)

    # Count white pixels
    area = np.count_nonzero(mask)

    return mask, area


def save_reference(label, image, mask, counters):

    if counters[label] >= MAX_PER_CLASS:
        return

    idx = counters[label]

    image_file = OUTPUT_REFRENCE_INSTANCE_BANK_ROOT / label / f"{label}_{idx:03d}_img.png"

    mask_file = OUTPUT_REFRENCE_INSTANCE_BANK_ROOT / label / f"{label}_{idx:03d}_mask.png"

    # save original image
    cv2.imwrite(str(image_file), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # save mask
    cv2.imwrite(str(mask_file), mask)

    counters[label] += 1


# ==========================================================
# Processing
# ==========================================================


def process_annotation(json_file, counters, split="train"):

    image_path = get_image_path(json_file, split)

    if not image_path.exists():
        print("Missing:", image_path)
        return

    image = load_image(image_path)

    if image is None:
        print("Cannot read:", image_path)
        return

    annotation = load_annotation(json_file)

    for obj in annotation["objects"]:

        valid, polygon = is_valid_object(obj)

        if not valid:
            continue

        label = obj["label"]

        # Already have enough references
        if counters[label] >= MAX_PER_CLASS:
            continue

        # Create mask
        mask, area = create_mask(image.shape, polygon)

        # Filter by actual white pixel area
        if area < MIN_AREA:
            continue

        save_reference(label, image, mask, counters)


# ==========================================================
# Main
# ==========================================================


def build_reference_instance_bank(split="train"):

    create_output_dirs()

    counters = {cls: 0 for cls in CLASSES}

    json_files = get_json_files(split)

    print(f"Found {len(json_files)} annotation files.")

    for idx, json_file in enumerate(json_files, start=1):

        process_annotation(json_file, counters, split)

        if idx % 100 == 0:
            print(f"{idx}/{len(json_files)} processed")

        # stop when every class has 11 samples
        if all(count >= MAX_PER_CLASS for count in counters.values()):
            break

    print("\nFinished\n")

    print("=" * 40)
    print("Reference bank summary")
    print("=" * 40)

    for cls in CLASSES:

        print(f"{cls:15s}: {counters[cls]}")


if __name__ == "__main__":

    build_reference_instance_bank("train")
