import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from config import *

# ==========================================================
# IO
# ==========================================================


def create_output_dirs():
    for cls in CLASSES:
        (OUTPUT_REFRENCE_SEMANTIC_BANK_ROOT / cls).mkdir(parents=True, exist_ok=True)


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
# Save reference
# ==========================================================


def save_reference(label, image, mask, counters):

    if counters[label] >= MAX_PER_CLASS:
        return

    idx = counters[label]

    image_file = OUTPUT_REFRENCE_SEMANTIC_BANK_ROOT / label / f"{label}_{idx:03d}_img.png"

    mask_file = OUTPUT_REFRENCE_SEMANTIC_BANK_ROOT / label / f"{label}_{idx:03d}_mask.png"

    cv2.imwrite(str(image_file), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    cv2.imwrite(str(mask_file), mask)

    counters[label] += 1


# ==========================================================
# Processing
# ==========================================================


def process_annotation(json_file, counters, split="train", min_instances=3):

    image_path = get_image_path(json_file, split)

    if not image_path.exists():
        return

    image = load_image(image_path)

    if image is None:
        return

    annotation = load_annotation(json_file)

    # ------------------------------------------------------
    # Group polygons by class
    # ------------------------------------------------------

    class_polygons = defaultdict(list)

    for obj in annotation["objects"]:

        label = obj["label"]

        if label not in CLASS_SET:
            continue

        polygon = np.asarray(obj["polygon"], dtype=np.int32)

        if len(polygon) < 3:
            continue

        class_polygons[label].append(polygon)

    h, w = image.shape[:2]

    # ------------------------------------------------------
    # Create one mask per class
    # ------------------------------------------------------

    for label, polygons in class_polygons.items():

        if counters[label] >= MAX_PER_CLASS:
            continue

        # Require multiple objects
        if len(polygons) < min_instances:
            continue

        mask = np.zeros((h, w), dtype=np.uint8)

        for polygon in polygons:

            cv2.fillPoly(mask, [polygon], 255)

        area = np.count_nonzero(mask)

        if area < MIN_AREA:
            continue

        save_reference(label, image, mask, counters)


# ==========================================================
# Main
# ==========================================================


def build_reference_bank(split="train", min_instances=3):

    create_output_dirs()

    counters = {cls: 0 for cls in CLASSES}

    json_files = get_json_files(split)

    print(f"Found {len(json_files)} annotation files.")

    for idx, json_file in enumerate(json_files, start=1):

        process_annotation(json_file, counters, split, min_instances)

        if idx % 100 == 0:

            print(f"{idx}/{len(json_files)} processed")

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
