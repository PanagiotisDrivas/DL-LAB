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
# Official Cityscapes label IDs
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


# ------------------------------------------------------------------
# Cityscapes thing classes
# ------------------------------------------------------------------

THING_CLASSES = {
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
}


# ------------------------------------------------------------------
# Polygon utilities
# ------------------------------------------------------------------

def polygon_area(polygon):
    """
    Compute polygon area using the Shoelace formula.

    Used to optionally draw larger polygons first so that smaller,
    more specific objects are drawn afterwards.
    """

    n = len(polygon)

    if n < 3:
        return 0.0

    area = 0.0

    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        area += x1 * y2 - x2 * y1

    return abs(area) / 2.0


def objects_largest_first(objects):
    """
    Sort objects by polygon area in descending order.
    """

    return sorted(
        objects,
        key=lambda obj: polygon_area(obj["polygon"]),
        reverse=True,
    )


# ------------------------------------------------------------------
# Palette image
# ------------------------------------------------------------------

def save_palette_image(mask, output_path):
    """
    Save a semantic mask using the Cityscapes palette.
    """

    img = Image.fromarray(mask, mode="P")

    palette = []

    for color in PALETTE:
        palette.extend(color)

    palette.extend(
        [0] * (256 * 3 - len(palette))
    )

    img.putpalette(palette)
    img.save(output_path)


# ------------------------------------------------------------------
# Semantic mask
# ------------------------------------------------------------------

def create_semantic_mask(json_file, output_path):
    """
    Create a semantic segmentation mask using the compact CLASS_IDS.
    """

    with open(json_file, "r") as f:
        data = json.load(f)

    height = data["imgHeight"]
    width = data["imgWidth"]

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    for obj in data["objects"]:

        label = obj["label"]

        if label not in CLASS_IDS:
            continue

        temp = Image.new(
            "L",
            (width, height),
            0,
        )

        draw = ImageDraw.Draw(temp)

        draw.polygon(
            obj["polygon"],
            fill=1,
        )

        binary = np.array(temp) > 0

        mask[binary] = CLASS_IDS[label]

    save_palette_image(
        mask,
        output_path,
    )

    print(
        "Saved semantic mask:",
        output_path.name,
        "Classes:",
        np.unique(mask),
    )


# ------------------------------------------------------------------
# Instance mask visualization
# ------------------------------------------------------------------

def instance_mask_to_color(instance_mask):
    """
    Convert an instance ID mask into a colorful RGB visualization.
    Each instance receives a deterministic color.
    """

    height, width = instance_mask.shape

    color_mask = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    instance_ids = np.unique(instance_mask)

    rng = np.random.default_rng(seed=42)

    colors = {}

    for instance_id in instance_ids:

        if instance_id == 0:
            colors[instance_id] = (0, 0, 0)
        else:
            colors[instance_id] = rng.integers(
                0,
                255,
                size=3,
            )

    for instance_id, color in colors.items():

        color_mask[
            instance_mask == instance_id
        ] = color

    return color_mask


# ------------------------------------------------------------------
# Compact instance mask
# ------------------------------------------------------------------

def create_instance_mask(json_file, output_path):
    """
    Create an instance mask using the compact CLASS_IDS.

    The raw instance ID mask is saved as:

        *_instances.png

    A colorful visualization is also saved as:

        *_instances_color.png
    """

    with open(json_file, "r") as f:
        data = json.load(f)

    height = data["imgHeight"]
    width = data["imgWidth"]

    instance_mask = np.zeros(
        (height, width),
        dtype=np.uint16,
    )

    class_instance_counter = {}

    for obj in data["objects"]:

        label = obj["label"]

        if label not in CLASS_IDS:
            continue

        class_id = CLASS_IDS[label]

        class_instance_counter[class_id] = (
            class_instance_counter.get(class_id, 0) + 1
        )

        instance_number = class_instance_counter[class_id]

        instance_id = (
            class_id * 1000
            + instance_number
        )

        temp = Image.new(
            "L",
            (width, height),
            0,
        )

        draw = ImageDraw.Draw(temp)

        draw.polygon(
            obj["polygon"],
            fill=1,
        )

        binary = np.array(temp) > 0

        instance_mask[binary] = instance_id

    # --------------------------------------------------
    # Color visualization
    # --------------------------------------------------

    color_output = Path(
        str(output_path).replace(
            ".png",
            "_instances.png",
        )
    )

    color_mask = instance_mask_to_color(
        instance_mask
    )

    Image.fromarray(
        color_mask,
        mode="RGB",
    ).save(color_output)

    print(
        "Saved color mask:",
        color_output.name,
    )

    print(
        "Instances:",
        np.unique(instance_mask),
    )


# ------------------------------------------------------------------
# Official Cityscapes labelIds mask
# ------------------------------------------------------------------

def create_labelids_mask(json_file, output_path):
    """
    Create a single-channel uint8 Cityscapes labelIds mask.

    Pixel values correspond to official Cityscapes label IDs.
    """

    with open(json_file, "r") as f:
        data = json.load(f)

    height = data["imgHeight"]
    width = data["imgWidth"]

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    for obj in data["objects"]:

        label = obj["label"]

        if label not in OFFICIAL_LABEL_IDS:
            continue

        temp = Image.new(
            "L",
            (width, height),
            0,
        )

        draw = ImageDraw.Draw(temp)

        draw.polygon(
            obj["polygon"],
            fill=1,
        )

        binary = np.array(temp) > 0

        mask[binary] = OFFICIAL_LABEL_IDS[label]

    Image.fromarray(
        mask,
        mode="L",
    ).save(output_path)

    print(
        "Saved labelIds:",
        output_path.name,
        "Classes:",
        np.unique(mask),
    )


# ------------------------------------------------------------------
# Official Cityscapes instanceIds mask
# ------------------------------------------------------------------

def create_instanceids_mask(json_file, output_path):
    """
    Create a single-channel uint16 Cityscapes instanceIds mask.

    Thing classes:
        labelId * 1000 + instance number

    Stuff classes:
        bare labelId
    """

    with open(json_file, "r") as f:
        data = json.load(f)

    height = data["imgHeight"]
    width = data["imgWidth"]

    mask = np.zeros(
        (height, width),
        dtype=np.uint16,
    )

    class_instance_counter = {}

    for obj in data["objects"]:

        label = obj["label"]

        if label not in OFFICIAL_LABEL_IDS:
            continue

        label_id = OFFICIAL_LABEL_IDS[label]

        # --------------------------------------------------
        # Thing classes get individual instance IDs
        # --------------------------------------------------

        if label in THING_CLASSES:

            class_instance_counter[label_id] = (
                class_instance_counter.get(
                    label_id,
                    0,
                ) + 1
            )

            instance_number = (
                class_instance_counter[label_id]
            )

            instance_id = (
                label_id * 1000
                + instance_number
            )

        # --------------------------------------------------
        # Stuff classes use the bare label ID
        # --------------------------------------------------

        else:

            instance_id = label_id

        temp = Image.new(
            "L",
            (width, height),
            0,
        )

        draw = ImageDraw.Draw(temp)

        draw.polygon(
            obj["polygon"],
            fill=1,
        )

        binary = np.array(temp) > 0

        mask[binary] = instance_id

    Image.fromarray(
        mask,
        mode="I;16",
    ).save(output_path)

    print(
        "Saved instanceIds:",
        output_path.name,
        "Instances:",
        np.unique(mask),
    )


# ------------------------------------------------------------------
# Rename annotation files
# ------------------------------------------------------------------

def rename_annotation_files(root_dir):
    """
    Rename:

        <name>_leftImg8bit.json

    to:

        <name>_gtFine_polygons.json
    """

    root_dir = Path(root_dir)

    for json_file in root_dir.rglob("*.json"):

        if "leftImg8bit" not in json_file.name:
            continue

        new_name = json_file.name.replace(
            "leftImg8bit",
            "gtFine_polygons",
        )

        new_path = (
            json_file.parent / new_name
        )

        json_file.rename(new_path)

        print(
            f"{json_file.name} -> {new_name}"
        )

    print(
        "Done renaming annotation files."
    )


# ------------------------------------------------------------------
# Process annotation folder
# ------------------------------------------------------------------

def process_folder(json_dir, output_dir):
    """
    Generate all masks from polygon JSON files.

    For:

        aachen_000166_000025_gtFine_polygons.json

    generate:

        aachen_000166_000025_gtFine_labelIds.png
        aachen_000166_000025_gtFine_instanceIds.png
        aachen_000166_000025_gtFine_instances.png
    """

    json_dir = Path(json_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_files = sorted(
        json_dir.glob(
            "*_gtFine_polygons.json"
        )
    )

    print(
        "Found JSON files:",
        len(json_files),
    )

    for json_file in json_files:

        # --------------------------------------------------
        # Remove "_gtFine_polygons"
        #
        # aachen_000166_000025_gtFine_polygons
        # ->
        # aachen_000166_000025
        # --------------------------------------------------

        base_name = json_file.stem.replace(
            "_gtFine_polygons",
            "",
        )

        # --------------------------------------------------
        # Compact instance mask
        #
        # create_instance_mask() adds "_instances"
        # --------------------------------------------------

        instance_output = (
            output_dir
            / f"{base_name}_gtFine.png"
        )

        create_instance_mask(
            json_file,
            instance_output,
        )

        # --------------------------------------------------
        # Official labelIds
        # --------------------------------------------------

        create_labelids_mask(
            json_file,
            output_dir
            / f"{base_name}_gtFine_labelIds.png",
        )

        # --------------------------------------------------
        # Official instanceIds
        # --------------------------------------------------

        create_instanceids_mask(
            json_file,
            output_dir
            / f"{base_name}_gtFine_instanceIds.png",
        )


# ------------------------------------------------------------------
# Organize Cityscapes structure
# ------------------------------------------------------------------

def organize_cityscapes_structure(
    gt_src,
    img_src,
    gt_root,
    img_root,
):
    """
    Copy polygon annotations, generated masks, and corresponding images.

    For each:

        *_gtFine_polygons.json

    copy:

        *_gtFine_polygons.json
        *_gtFine_labelIds.png
        *_gtFine_instanceIds.png
        *_gtFine_instances.png

    Images are copied only when a corresponding annotation exists.
    """

    gt_src = Path(gt_src)
    img_src = Path(img_src)

    gt_root = Path(gt_root)
    img_root = Path(img_root)

    # --------------------------------------------------
    # Copy GT annotations and masks
    # --------------------------------------------------

    print(
        "\nCopying GT annotations and masks..."
    )

    gt_files = set()

    polygon_files = sorted(
        gt_src.glob(
            "*_gtFine_polygons.json"
        )
    )

    for polygon_file in polygon_files:

        # Example:
        #
        # aachen_000166_000025_gtFine_polygons.json
        #
        # ->
        #
        # aachen_000166_000025

        base_name = polygon_file.name.replace(
            "_gtFine_polygons.json",
            "",
        )

        city = polygon_file.name.split(
            "_"
        )[0]

        dst_dir = gt_root / city

        dst_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # --------------------------------------------------
        # Copy polygon JSON
        # --------------------------------------------------

        shutil.copy2(
            polygon_file,
            dst_dir / polygon_file.name,
        )

        print(
            f"Copied GT: {polygon_file.name}"
        )

        # --------------------------------------------------
        # Copy generated masks
        # --------------------------------------------------

        mask_files = [
            gt_src
            / f"{base_name}_gtFine_labelIds.png",

            gt_src
            / f"{base_name}_gtFine_instanceIds.png",

            gt_src
            / f"{base_name}_gtFine_instances.png",
        ]

        for mask_file in mask_files:

            if not mask_file.exists():

                print(
                    f"Warning: mask not found: "
                    f"{mask_file.name}"
                )

                continue

            shutil.copy2(
                mask_file,
                dst_dir / mask_file.name,
            )

            print(
                f"Copied mask: "
                f"{mask_file.name}"
            )

        # Used for image matching
        gt_files.add(base_name)

    # --------------------------------------------------
    # Copy only images with GT
    # --------------------------------------------------

    print(
        "\nCopying images with GT..."
    )

    for file in img_src.iterdir():

        if file.name.startswith("."):
            continue

        if file.suffix.lower() not in [
            ".png",
            ".jpg",
            ".jpeg",
        ]:
            continue

        image_key = file.stem.replace(
            "_leftImg8bit",
            "",
        )

        # --------------------------------------------------
        # Skip images without annotations
        # --------------------------------------------------

        if image_key not in gt_files:

            print(
                f"Skipping image without GT: "
                f"{file.name}"
            )

            continue

        city = file.name.split(
            "_"
        )[0]

        dst_dir = img_root / city

        dst_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            file,
            dst_dir / file.name,
        )

        print(
            f"Copied image: {file.name}"
        )

    print(
        "\nCityscapes dataset copy completed."
    )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--json-dir",
        default="input/annotations",
        help="Directory containing annotation JSON files",
    )

    parser.add_argument(
        "--output-dir",
        default="input/annotations",
        help="Directory to save generated masks",
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # Step 1: Rename JSON files
    # --------------------------------------------------

    rename_annotation_files(
        args.json_dir
    )

    # --------------------------------------------------
    # Step 2: Generate masks
    # --------------------------------------------------

    process_folder(
        json_dir=args.json_dir,
        output_dir=args.output_dir,
    )

    # --------------------------------------------------
    # Step 3: Organize Cityscapes structure
    # --------------------------------------------------

    organize_cityscapes_structure(
        gt_src="input/annotations",
        img_src="input/images",
        gt_root="insid3_dataset/gtFine/train",
        img_root="insid3_dataset/leftImg8bit_sequence/train",
    )