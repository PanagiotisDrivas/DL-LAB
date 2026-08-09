import argparse
import re
import subprocess
from pathlib import Path
from utils.merge import merge_json
from create_mask import process_folder, copy_first_mask_ground_truth
from config import CLASSES
import shutil

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model-name",
    default="dinov3_vits16"
)

parser.add_argument(
    "--input-dir",
    default="aachen"
)


parser.add_argument(
    "--topk",
    default="5"
)

parser.add_argument(
    "--max-context-length",
    default="11"
)

parser.add_argument(
    "--short-side",
    # default="1024"
    default="768"   
)

parser.add_argument(
    "--temperature",
    default="0.01",
    help="Softmax temperature in propagate(); lower = sharper/more decisive class "
    "assignment, higher = smoother blending across matched classes",
)

parser.add_argument(
    "--num-before-frames",
    default="19",
    help="How many frames in frames/ come chronologically before the GT frame",
)

parser.add_argument(
    "--dataset-root",
    default="temporal_dataset",
    help="Root of the SPINO-ready dataset tree; labelIds/instanceIds are written to "
    "<dataset-root>/gtFine/<split>/<input-dir>/",
)
parser.add_argument(
    "--split",
    default="train",
    help="Split subfolder under gtFine/ and leftImg8bit_sequence/ (e.g. train, val)",
)

args = parser.parse_args()


for class_name in CLASSES:

    print("==============================")
    print("Processing:", class_name)
    print("==============================")


    # Create instance mask

    subprocess.run([
        "python",
        "instance_segmentation/create_instance_pipeline.py",
        "--input-json",
        f"input/{args.input_dir}/first_mask/",
        "--class-name",
        class_name,
        "--output-dir",
        f"output/{args.input_dir}/probs/instance_mask/{class_name}_output"
    ])


    # DINOv3 tracking

    subprocess.run([
        "python",
        "dinov3_seg_tracking.py",

        "--dinov3-location",
        ".",

        "--model-name",
        args.model_name,

        "--frames-dir",
        f"input/{args.input_dir}/frames",

        "--first-mask",
        f"output/{args.input_dir}/probs/instance_mask/{class_name}_output/{class_name}_instance_mask.png",

        "--reference-dir",
        f"input/{args.input_dir}/first_mask",

        "--num-before-frames",
        args.num_before_frames,

        "--output-dir",
        f"output/{args.input_dir}/probs/out/{args.model_name}_{args.short_side}/{class_name}_tracking_{args.model_name}_{args.short_side}",

        "--short-side",
        args.short_side,

        "--max-context-length",
        args.max_context_length,

        "--topk",
        args.topk,

        "--temperature",
        args.temperature
    ])

merge_json(
    input_dir=f"output/{args.input_dir}/probs/out/{args.model_name}_{args.short_side}/",
    model_name=args.model_name,
    short_side=args.short_side,
    frames_dir=f"input/{args.input_dir}/frames",
    output_dir=f"input/{args.input_dir}/annotations/"
)

# Render labelIds.png / instanceIds.png (gtFine-named) straight into the
# SPINO-ready dataset tree. Multiple input dirs can be different sequences of the
# same city (e.g. bremen1/bremen2, darmstadt1/2/3) -- strip the trailing sequence
# digit so they all land in one shared <city>/ folder, matching SPINO's layout,
# instead of each creating its own separate (and never-checked) output folder.
city_name = re.sub(r"\d+$", "", args.input_dir)
gt_fine_dir = Path(args.dataset_root) / "gtFine" / args.split / city_name
process_folder(
    json_dir=f"input/{args.input_dir}/annotations",
    output_dir=str(gt_fine_dir),
)

copy_first_mask_ground_truth(
    first_mask_dir=f"input/{args.input_dir}/first_mask",
    output_dir=str(gt_fine_dir),
)

### Delete intermediate_results
shutil.rmtree(
    f"output/{args.input_dir}",
    ignore_errors=True,
)
