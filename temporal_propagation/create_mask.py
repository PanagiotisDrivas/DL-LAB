import json
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


def save_palette_image(mask, output_path):

    img = Image.fromarray(
        mask,
        mode="P"
    )

    palette = []

    for color in PALETTE:
        palette.extend(color)

    palette.extend(
        [0] * (256 * 3 - len(palette))
    )

    img.putpalette(palette)

    img.save(output_path)



def create_semantic_mask(
        json_file,
        output_path
):

    with open(json_file, "r") as f:
        data = json.load(f)


    height = data["imgHeight"]
    width = data["imgWidth"]


    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )


    for obj in data["objects"]:

        label = obj["label"]


        if label not in CLASS_IDS:
            continue


        temp = Image.new(
            "L",
            (width, height),
            0
        )


        draw = ImageDraw.Draw(temp)


        draw.polygon(
            obj["polygon"],
            fill=1
        )


        binary = np.array(temp) > 0


        mask[binary] = CLASS_IDS[label]



    save_palette_image(
        mask,
        output_path
    )


    print(
        "Saved:",
        output_path.name,
        "Classes:",
        np.unique(mask)
    )



def process_folder(
        json_dir,
        output_dir
):

    json_dir = Path(json_dir)
    output_dir = Path(output_dir)


    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    json_files = sorted(
        json_dir.glob("*.json")
    )


    print(
        "Found JSON files:",
        len(json_files)
    )


    for json_file in json_files:

        output_path = (
            output_dir
            /
            f"{json_file.stem}.png"
        )


        create_semantic_mask(
            json_file,
            output_path
        )



if __name__ == "__main__":

    import argparse


    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--json-dir",
        required=True,
        help="Directory containing annotation JSON files"
    )


    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save semantic masks"
    )


    args = parser.parse_args()


    process_folder(
        json_dir=args.json_dir,
        output_dir=args.output_dir
    )