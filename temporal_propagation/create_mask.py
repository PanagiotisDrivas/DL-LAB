import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# ------------------------------------------------------------------
# Semantic class IDs
# ------------------------------------------------------------------

CLASS_IDS = {
    "unlabeled": 0,
    "road": 1,
    "sidewalk": 2,
    "building": 3,
    "wall": 4,
    "fence": 5,
    "pole": 6,
    "traffic light": 7,
    "traffic sign": 8,
    "vegetation": 9,
    "terrain": 10,
    "sky": 11,
    "person": 12,
    "rider": 13,
    "car": 14,
    "truck": 15,
    "bus": 16,
    "train": 17,
    "motorcycle": 18,
    "bicycle": 19,
}


# ------------------------------------------------------------------
# Cityscapes palette
# ------------------------------------------------------------------

PALETTE = [
    (0, 0, 0),
    (128, 64, 128),
    (244, 35, 232),
    (70, 70, 70),
    (102, 102, 156),
    (190, 153, 153),
    (153, 153, 153),
    (250, 170, 30),
    (220, 220, 0),
    (107, 142, 35),
    (152, 251, 152),
    (70, 130, 180),
    (220, 20, 60),
    (255, 0, 0),
    (0, 0, 142),
    (0, 0, 70),
    (0, 60, 100),
    (0, 80, 100),
    (0, 0, 230),
    (119, 11, 32),
]

# ------------------------------------------------------------------
# Official Cityscapes labelIds (not this pipeline's compact CLASS_IDS)
# https://github.com/mcordts/cityscapesScripts - labels.py
# Used for labelIds.png / instanceIds.png so they line up pixel-value-wise
# with the reference gtFine files in input/{city}/first_mask/rest/.
# ------------------------------------------------------------------

OFFICIAL_LABEL_IDS = {
    "road": 7,
    "sidewalk": 8,
    "building": 11,
    "wall": 12,
    "fence": 13,
    "pole": 17,
    "traffic light": 19,
    "traffic sign": 20,
    "vegetation": 21,
    "terrain": 22,
    "sky": 23,
    "person": 24,
    "rider": 25,
    "car": 26,
    "truck": 27,
    "bus": 28,
    "train": 31,
    "motorcycle": 32,
    "bicycle": 33,
}

# "thing" classes get per-object instance IDs (labelId * 1000 + n) in instanceIds.png;
# "stuff" classes (the rest) just carry their bare labelId, same as in labelIds.png.
THING_CLASSES = {"person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"}

# Official per-class Cityscapes colors, keyed by label name (same order as
# OFFICIAL_LABEL_IDS and PALETTE, so every class shares one color regardless of
# instance -- matching the real *_gtFine_color.png ground truth files exactly,
# for direct visual comparison.
OFFICIAL_LABEL_COLORS = dict(zip(OFFICIAL_LABEL_IDS.keys(), PALETTE[1:]))


def save_palette_image(mask, output_path):

    img = Image.fromarray(mask, mode="P")

    palette = []

    for color in PALETTE:
        palette.extend(color)

    palette.extend([0] * (256 * 3 - len(palette)))

    img.putpalette(palette)

    img.save(output_path)


def create_semantic_mask(json_file, output_path):

    with open(json_file, "r") as f:
        data = json.load(f)

    height = data["imgHeight"]
    width = data["imgWidth"]

    mask = np.zeros((height, width), dtype=np.uint8)

    for obj in data["objects"]:
    # for obj in objects_largest_first(data["objects"]):

        label = obj["label"]

        if label not in CLASS_IDS:
            continue

        temp = Image.new("L", (width, height), 0)

        draw = ImageDraw.Draw(temp)

        draw.polygon([tuple(pt) for pt in obj["polygon"]], fill=1)

        binary = np.array(temp) > 0

        mask[binary] = CLASS_IDS[label]

    save_palette_image(mask, output_path)

    print("Saved:", output_path.name, "Classes:", np.unique(mask))

def instance_mask_to_color(instance_mask):
    """
    Convert instance ID mask to a colorful RGB visualization.
    Each instance gets a unique color.
    """

    height, width = instance_mask.shape

    color_mask = np.zeros((height, width, 3), dtype=np.uint8)

    instance_ids = np.unique(instance_mask)

    rng = np.random.default_rng(seed=42)

    colors = {}

    for instance_id in instance_ids:

        if instance_id == 0:
            colors[instance_id] = (0, 0, 0)  # background
        else:
            colors[instance_id] = rng.integers(
                0, 255, size=3
            )

    for instance_id, color in colors.items():
        color_mask[instance_mask == instance_id] = color

    return color_mask
def create_instance_mask(json_file, output_path):

    with open(json_file, "r") as f:
        data = json.load(f)

    height = data["imgHeight"]
    width = data["imgWidth"]

    instance_mask = np.zeros(
        (height, width),
        dtype=np.uint16
    )

    class_instance_counter = {}

    for obj in data["objects"]:
    # for obj in objects_largest_first(data["objects"]):

        label = obj["label"]

        if label not in CLASS_IDS:
            continue

        class_id = CLASS_IDS[label]

        class_instance_counter[class_id] = (
            class_instance_counter.get(class_id, 0) + 1
        )

        instance_number = class_instance_counter[class_id]

        instance_id = class_id * 1000 + instance_number

        temp = Image.new(
            "L",
            (width, height),
            0
        )

        draw = ImageDraw.Draw(temp)

        draw.polygon(
            [tuple(pt) for pt in obj["polygon"]],
            fill=1
        )

        binary = np.array(temp) > 0

        instance_mask[binary] = instance_id


    # -------- Save raw instance IDs --------
    instance_output = Path(
        str(output_path).replace(
            ".png",
            "_instances.png"
        )
    )

    Image.fromarray(
        instance_mask,
        mode="I;16"
    ).save(instance_output)


    # -------- Save colorful visualization --------
    color_output = Path(
        str(output_path).replace(
            ".png",
            "_instances.png"
        )
    )

    color_mask = instance_mask_to_color(instance_mask)

    Image.fromarray(
        color_mask,
        mode="RGB"
    ).save(color_output)


    print("Saved ID mask:", instance_output.name)
    print("Saved color mask:", color_output.name)
    print("Instances:", np.unique(instance_mask))


def create_labelids_mask(json_file, output_path):
    """Single-channel uint8 PNG, pixel value = official Cityscapes labelId
    (matches input/{city}/first_mask/rest/*_gtFine_labelIds.png)."""

    with open(json_file, "r") as f:
        data = json.load(f)

    height = data["imgHeight"]
    width = data["imgWidth"]

    mask = np.zeros((height, width), dtype=np.uint8)

    for obj in data["objects"]:
    # for obj in objects_largest_first(data["objects"]):

        label = obj["label"]

        if label not in OFFICIAL_LABEL_IDS:
            continue

        temp = Image.new("L", (width, height), 0)

        draw = ImageDraw.Draw(temp)

        draw.polygon([tuple(pt) for pt in obj["polygon"]], fill=1)

        binary = np.array(temp) > 0

        mask[binary] = OFFICIAL_LABEL_IDS[label]

    Image.fromarray(mask, mode="L").save(output_path)

    print("Saved labelIds:", output_path.name, "Classes:", np.unique(mask))


def create_instanceids_mask(json_file, output_path):
    """Single-channel uint16 PNG, pixel value = labelId * 1000 + instance number
    (matches input/{city}/first_mask/rest/*_gtFine_instanceIds.png)."""

    with open(json_file, "r") as f:
        data = json.load(f)

    height = data["imgHeight"]
    width = data["imgWidth"]

    mask = np.zeros((height, width), dtype=np.uint16)

    class_instance_counter = {}
    
    for obj in data["objects"]:
    # for obj in objects_largest_first(data["objects"]):

        label = obj["label"]

        if label not in OFFICIAL_LABEL_IDS:
            continue

        label_id = OFFICIAL_LABEL_IDS[label]

        if label in THING_CLASSES:
            class_instance_counter[label_id] = class_instance_counter.get(label_id, 0) + 1
            instance_number = class_instance_counter[label_id]
            instance_id = label_id * 1000 + instance_number
        else:
            instance_id = label_id

        temp = Image.new("L", (width, height), 0)

        draw = ImageDraw.Draw(temp)

        draw.polygon([tuple(pt) for pt in obj["polygon"]], fill=1)

        binary = np.array(temp) > 0

        mask[binary] = instance_id

    Image.fromarray(mask, mode="I;16").save(output_path)

    print("Saved instanceIds:", output_path.name, "Instances:", np.unique(mask))


def create_gtfine_color_mask(json_file, output_path):
    """RGB PNG colored with the official per-class Cityscapes palette (same color
    for every instance of a class), matching the real *_gtFine_color.png ground
    truth files pixel-for-pixel in color scheme, for direct visual comparison."""

    with open(json_file, "r") as f:
        data = json.load(f)

    height = data["imgHeight"]
    width = data["imgWidth"]

    color_mask = np.zeros((height, width, 3), dtype=np.uint8)

    for obj in data["objects"]:

        label = obj["label"]

        if label not in OFFICIAL_LABEL_COLORS:
            continue

        temp = Image.new("L", (width, height), 0)

        draw = ImageDraw.Draw(temp)

        draw.polygon([tuple(pt) for pt in obj["polygon"]], fill=1)

        binary = np.array(temp) > 0

        color_mask[binary] = OFFICIAL_LABEL_COLORS[label]

    Image.fromarray(color_mask, mode="RGB").save(output_path)

    print("Saved color (Cityscapes palette):", output_path.name)


def gt_fine_stem(json_stem: str) -> str:
    """<city>_<seq>_<frame>_leftImg8bit -> <city>_<seq>_<frame>_gtFine, or
    <city>_<seq>_<frame>_gtFine_polygons -> <city>_<seq>_<frame>_gtFine (already
    gtFine-named annotation, just drop "_polygons"), matching the naming SPINO's
    dataset loader expects for labelIds/instanceIds files."""
    if json_stem.endswith("_leftImg8bit"):
        return json_stem[: -len("_leftImg8bit")] + "_gtFine"
    if json_stem.endswith("_gtFine_polygons"):
        return json_stem[: -len("_polygons")]
    if json_stem.endswith("_gtFine"):
        return json_stem
    return f"{json_stem}_gtFine"


def copy_first_mask_ground_truth(first_mask_dir, output_dir):
    """Copies the human-annotated GT frame's polygons.json (in first_mask_dir) and its
    labelIds/instanceIds/color PNGs (in first_mask_dir/rest) into the gtFine dataset
    tree, alongside the propagated frames' masks. Filenames already match the
    <city>_<seq>_<frame>_gtFine_{polygons.json,labelIds.png,instanceIds.png,color.png}
    convention, so no renaming is needed."""

    first_mask_dir = Path(first_mask_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    for gt_file in first_mask_dir.glob("*_gtFine_polygons.json"):
        shutil.copy2(gt_file, output_dir / gt_file.name)

    rest_dir = first_mask_dir / "rest"

    for suffix in ("_gtFine_labelIds.png", "_gtFine_instanceIds.png", "_gtFine_color.png"):
        for gt_file in rest_dir.glob(f"*{suffix}"):
            shutil.copy2(gt_file, output_dir / gt_file.name)


def process_folder(json_dir, output_dir):

    json_dir = Path(json_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(json_dir.glob("*.json"))

    print("Found JSON files:", len(json_files))

    for json_file in json_files:

        # create_semantic_mask(json_file, output_path)
        # create_instance_mask(json_file, output_path)  # superseded by create_gtfine_color_mask

        gt_stem = gt_fine_stem(json_file.stem)

        create_labelids_mask(
            json_file,
            output_dir / f"{gt_stem}_labelIds.png"
        )

        create_instanceids_mask(
            json_file,
            output_dir / f"{gt_stem}_instanceIds.png"
        )

        create_gtfine_color_mask(
            json_file,
            output_dir / f"{gt_stem}_color.png"
        )

        shutil.copy2(
            json_file,
            output_dir / f"{gt_stem}_polygons.json"
        )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--json-dir", required=True, help="Directory containing annotation JSON files"
    )

    parser.add_argument(
        "--output-dir", required=True, help="Directory to save semantic masks"
    )

    args = parser.parse_args()

    process_folder(json_dir=args.json_dir, output_dir=args.output_dir)
