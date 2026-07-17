import argparse
from pathlib import Path
import sys
import json

import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import label
import json

# allow imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from instance_segmentation.class_definitions import (
    CLASS_IDS,
    ID_TO_CLASS,
)


MIN_PIXELS = 200


def random_color(seed):

    rng = np.random.default_rng(seed)

    return rng.integers(
        50,
        255,
        size=3,
        dtype=np.uint8
    )

def polygon_to_mask(polygon, height, width):

    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    pts = np.array(
        polygon,
        dtype=np.int32
    )

    if pts.shape[0] < 3:
        return mask

    pts = pts.reshape(
        (-1, 1, 2)
    )

    cv2.fillPoly(
        mask,
        [pts],
        1
    )

    return mask

def mask_to_polygon(mask):

    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    polygons = []

    for contour in contours:

        if len(contour) < 3:
            continue

        polygon = contour.reshape(-1, 2).tolist()

        polygons.append(
            polygon
        )

    return polygons


def save_json(
        instance_mask,
        class_id,
        output_json
):

    H, W = instance_mask.shape

    objects = []

    ids = np.unique(instance_mask)

    ids = ids[ids != 0]


    for instance_id in ids:

        binary = instance_mask == instance_id

        polygons = mask_to_polygon(binary)

        if len(polygons) == 0:
            continue


        objects.append(
            {
                "instanceId": int(instance_id),
                "classId": int(class_id),
                "label": ID_TO_CLASS.get(
                    int(class_id),
                    "unknown"
                ),
                "area": int(binary.sum()),
                "polygon": polygons,
            }
        )


    data = {
        "imgHeight": int(H),
        "imgWidth": int(W),
        "objects": objects,
    }


    with open(output_json, "w") as f:
        json.dump(
            data,
            f,
            indent=2
        )

def create_pipeline(
        mask_dir,
        class_name,
        output_dir
):

    output_dir = Path(output_dir)
    mask_dir = Path(mask_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    annotation_json = list(mask_dir.glob("*.json"))[0]


    # ----------------------------
    # Load JSON
    # ----------------------------

    with open(annotation_json, "r") as f:
        data = json.load(f)


    H = data["imgHeight"]
    W = data["imgWidth"]


    print("Loaded JSON")
    print("shape:", H, W)
    print("class:", class_name)


    # ----------------------------
    # Extract class polygons
    # ----------------------------

    objects = []

    for obj in data["objects"]:

        if obj.get("label") == class_name:

            objects.append(obj)


    print(
        "Found objects:",
        len(objects)
    )


    if len(objects) == 0:
        print("No objects found")
        return


    instance_mask = np.zeros(
        (H, W),
        dtype=np.uint16
    )


    visualization = np.zeros(
        (
            H,
            W,
            3
        ),
        dtype=np.uint8
    )


    valid = 0


    class_id = CLASS_IDS[class_name]


    # ----------------------------
    # Create instance mask
    # ----------------------------

    for obj in objects:

        valid += 1

        instance_id = (
            class_id * 1000
            +
            valid
        )

        mask = polygon_to_mask(
            obj["polygon"],
            H,
            W
        )


        area = int(mask.sum())


        print(
            f"{valid}: "
            f"id={instance_id}, "
            f"area={area}"
        )


        if area == 0:
            print("WARNING: empty polygon")
            continue


        instance_mask[mask == 1] = instance_id


        visualization[mask == 1] = random_color(
            instance_id
        )

    # ----------------------------
    # Save instance mask
    # ----------------------------

    Image.fromarray(
        instance_mask
    ).save(
        output_dir /
        f"{class_name}_instance_mask.png"
    )


    # ----------------------------
    # Save visualization
    # ----------------------------

    Image.fromarray(
        visualization
    ).save(
        output_dir /
        f"{class_name}_visualization.png"
    )


    # ----------------------------
    # Save reconstructed JSON
    # ----------------------------

    new_objects = []

    for obj_id, obj in enumerate(objects, start=1):

        new_objects.append(
            {
                "instanceId": class_id * 1000 + obj_id,
                "classId": class_id,
                "label": class_name,
                "area": int(mask.sum()),
                "polygon": obj["polygon"]
            }
        )


    output_json = output_dir / f"{class_name}_instances.json"


    with open(output_json, "w") as f:
        json.dump(
            {
                "imgHeight": H,
                "imgWidth": W,
                "objects": new_objects
            },
            f,
            indent=2
        )


    print("\nDONE")
    print("Objects:", valid)
    print("Mask:", output_dir / f"{class_name}_instance_mask.png")
    print("JSON:", output_json)


    # ----------------------------
    # Save outputs
    # ----------------------------

    Image.fromarray(
        instance_mask
    ).save(
        output_dir /
        f"{class_name}_instance_mask.png"
    )


    Image.fromarray(
        visualization
    ).save(
        output_dir /
        f"{class_name}_visualization.png"
    )


    print("\nDONE")
    print("Objects:", valid)
    print("Output:", output_dir)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--input-json",
        required=True
    )


    parser.add_argument(
        "--class-name",
        required=True
    )


    parser.add_argument(
        "--output-dir",
        required=True
    )


    args = parser.parse_args()


    create_pipeline(
        args.input_json,
        args.class_name,
        args.output_dir
    )