import json
import shutil
from pathlib import Path
from config import CLASSES

def get_frame_mapping(frames_dir):

    frames = sorted(
        list(Path(frames_dir).glob("*.png")) +
        list(Path(frames_dir).glob("*.jpg"))
    )

    mapping = {}

    for idx, frame in enumerate(frames):

        file_base = frame.stem

        if file_base.endswith("_leftImg8bit"):
            file_base = file_base[: -len("_leftImg8bit")]

        # matches Cityscapes' own "_gtFine_polygons.json" annotation naming, so a
        # re-run overwrites the same file in place instead of writing alongside it
        # under a different name
        mapping[idx] = f"{file_base}_gtFine_polygons.json"

    print("Frame mapping:")

    for k, v in list(mapping.items())[:5]:
        print(
            k,
            "->",
            v
        )

    return mapping



def rename_to_frame_names(
        tracking_json_dir,
        frames_dir,
        output_dir
):
    """Copies each positional {i:06d}.json from a single joint (all-classes)
    tracking run to its real frame filename. Unlike merge_json, there's nothing
    to combine here since one joint tracking run already contains every class."""

    tracking_json_dir = Path(tracking_json_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    json_files = sorted(tracking_json_dir.glob("*.json"))

    print("Tracking frames:", len(json_files))

    frame_mapping = get_frame_mapping(frames_dir)

    for json_file in json_files:

        frame_id = int(json_file.stem)

        output_name = frame_mapping.get(frame_id, json_file.name)

        shutil.copy(json_file, output_dir / output_name)

        print(f"Copied {json_file.name} -> {output_name}")


def merge_json(
        input_dir,
        model_name,
        short_side,
        frames_dir,
        output_dir
):

    input_dir = Path(input_dir)


    tracking_dirs = []


    for class_name in CLASSES:

        folder = (
            input_dir
            /
            f"{class_name}_tracking_{model_name}_{short_side}"
            /
            "json"
        )


        if folder.exists():

            tracking_dirs.append(folder)



    if len(tracking_dirs) == 0:

        raise RuntimeError(
            "No tracking JSON folders found"
        )



    print(
        "Found tracking folders:",
        len(tracking_dirs)
    )



    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )



    # Get frame jsons

    frames = sorted(
        tracking_dirs[0].glob("*.json")
    )


    print(
        "Tracking frames:",
        len(frames)
    )



    frame_mapping = get_frame_mapping(
        frames_dir
    )



    for frame in frames:


        objects = []

        img_height = None
        img_width = None



        for folder in tracking_dirs:


            json_file = folder / frame.name


            if not json_file.exists():
                continue



            with open(
                json_file,
                "r"
            ) as f:

                data = json.load(f)



            img_height = data["imgHeight"]

            img_width = data["imgWidth"]



            objects.extend(
                data["objects"]
            )



        # 000000.json -> index 0

        frame_id = int(
            frame.stem
        )



        output_name = frame_mapping.get(
            frame_id,
            frame.name
        )



        merged = {

            "imgHeight": img_height,

            "imgWidth": img_width,

            "objects": objects

        }



        with open(
            output_dir / output_name,
            "w"
        ) as f:

            json.dump(
                merged,
                f,
                indent=2
            )



        print(
            f"Merged {frame.name} -> {output_name} "
            f"objects={len(objects)}"
        )



if __name__ == "__main__":

    import argparse


    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--input-dir",
        required=True
    )


    parser.add_argument(
        "--model-name",
        default="dinov3_vits16"
    )


    parser.add_argument(
        "--short-side",
        default="512"
    )


    parser.add_argument(
        "--frames-dir",
        required=True
    )


    parser.add_argument(
        "--output-dir",
        required=True
    )


    args = parser.parse_args()



    merge_json(
        input_dir=args.input_dir,
        model_name=args.model_name,
        short_side=args.short_side,
        frames_dir=args.frames_dir,
        output_dir=args.output_dir
    )