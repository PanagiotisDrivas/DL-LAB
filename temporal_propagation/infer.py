import argparse
import subprocess
from utils.merge import merge_json
import shutil
from config import CLASSES

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
    default="768"
)

parser.add_argument(
    "--num-before-frames",
    default="19",
    help="How many frames in frames/ come chronologically before the GT frame",
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
        args.topk
    ])

merge_json(
    input_dir=f"output/{args.input_dir}/probs/out/{args.model_name}_{args.short_side}/",
    model_name=args.model_name,
    short_side=args.short_side,
    frames_dir=f"input/{args.input_dir}/frames",
    output_dir=f"input/{args.input_dir}/annotations/"
)

### Delete intermediate_results
shutil.rmtree(
    f"output/{args.input_dir}",
    ignore_errors=True,
)