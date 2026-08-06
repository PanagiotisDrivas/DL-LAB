import argparse
import re
import subprocess
from pathlib import Path
from utils.merge import rename_to_frame_names
from create_mask import process_folder
import shutil

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model-name",
    default="dinov3_vitb16"
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
    default="1024"
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


# Build one combined instance mask across every class, instead of one class at a
# time, so DINOv3 tracking can run as a single joint multi-class propagation
# where every pixel's label is decided by real competition between all classes
# at once (rather than 19 independent this-class-vs-background trackers pasted
# together afterward with no cross-class competition). Also ~19x fewer DINOv3
# forward passes per frame.

subprocess.run([
    "python",
    "instance_segmentation/create_instance_pipeline.py",
    "--input-json",
    f"input/{args.input_dir}/first_mask/",
    "--all-classes",
    "--output-dir",
    f"output/{args.input_dir}/probs/instance_mask/all_output"
])


# DINOv3 tracking (single joint run across all classes)

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
    f"output/{args.input_dir}/probs/instance_mask/all_output/combined_instance_mask.png",

    "--reference-dir",
    f"input/{args.input_dir}/first_mask",

    "--num-before-frames",
    args.num_before_frames,

    "--output-dir",
    f"output/{args.input_dir}/probs/out/{args.model_name}_{args.short_side}/tracking_{args.model_name}_{args.short_side}",

    "--short-side",
    args.short_side,

    "--max-context-length",
    args.max_context_length,

    "--topk",
    args.topk,

    "--temperature",
    args.temperature
])

rename_to_frame_names(
    tracking_json_dir=f"output/{args.input_dir}/probs/out/{args.model_name}_{args.short_side}/tracking_{args.model_name}_{args.short_side}/json",
    frames_dir=f"input/{args.input_dir}/frames",
    output_dir=f"input/{args.input_dir}/annotations/"
)

# Render labelIds.png / instanceIds.png (gtFine-named) straight into the
# SPINO-ready dataset tree. Multiple input dirs can be different sequences of the
# same city (e.g. bremen1/bremen2, darmstadt1/2/3) -- strip the trailing sequence
# digit so they all land in one shared <city>/ folder, matching SPINO's layout,
# instead of each creating its own separate (and never-checked) output folder.
city_name = re.sub(r"\d+$", "", args.input_dir)
process_folder(
    json_dir=f"input/{args.input_dir}/annotations",
    output_dir=str(Path(args.dataset_root) / "gtFine" / args.split / city_name),
)

### Delete intermediate_results
shutil.rmtree(
    f"output/{args.input_dir}",
    ignore_errors=True,
)
