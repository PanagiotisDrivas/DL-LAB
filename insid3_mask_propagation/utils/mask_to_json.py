import cv2
import json
import numpy as np
from pathlib import Path


def masks_to_json(mask_root, output_json):
    """
    mask_root:
        output/{img}_model/
            car.png
            building.png
            sky.png

    output_json:
        final json file
    """

    objects = []

    img_height = None
    img_width = None

    mask_root = Path(mask_root)

    # Each class mask file
    for mask_file in mask_root.glob("*.png"):

        # class name from filename
        label = mask_file.stem
        # example:
        # car.png -> car
        # building.png -> building

        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            continue

        if img_height is None:
            img_height, img_width = mask.shape

        # Convert white regions to binary
        binary = (mask > 127).astype(np.uint8)

        # Find separate instances
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:

            # remove small noise
            if cv2.contourArea(contour) < 10:
                continue

            polygon = contour.squeeze().tolist()

            # invalid polygon
            if len(polygon) < 3:
                continue

            objects.append({"label": label, "polygon": polygon})

    result = {"imgHeight": img_height, "imgWidth": img_width, "objects": objects}

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w") as f:
        json.dump(result, f, indent=4)

    print(f"Saved: {output_json}")
    print(f"Objects found: {len(objects)}")
