import argparse
import subprocess
from pathlib import Path
import shutil
from utils.merge import merge_json
from config import CLASSES

parser = argparse.ArgumentParser()

parser.add_argument("--model-name", default="dinov3_vitb16")
parser.add_argument("--topk", default="5")
parser.add_argument("--max-context-length", default="11")
parser.add_argument("--short-side", default="768")

args = parser.parse_args()

# Process every dataset inside input/
input_root = Path("input")
datasets = sorted([d for d in input_root.iterdir() if d.is_dir()])

for dataset in datasets:

    dataset_name = dataset.name

    print("\n" + "=" * 60)
    print(f"Processing dataset: {dataset_name}")
    print("=" * 60)

    for class_name in CLASSES:

        print(f"\nProcessing class: {class_name}")

        try:
            # Create instance mask
            subprocess.run([
                "python",
                "instance_segmentation/create_instance_pipeline.py",
                "--input-json",
                str(dataset / "first_mask"),
                "--class-name",
                class_name,
                "--output-dir",
                f"output/{dataset_name}/probs/instance_mask/{class_name}_output",
            ], check=True)
        except:
            pass

        try:
            # DINOv3 tracking
            subprocess.run([
                "python",
                "dinov3_seg_tracking.py",
                "--dinov3-location", ".",
                "--model-name", args.model_name,
                "--frames-dir", str(dataset / "frames"),
                "--first-mask",
                f"output/{dataset_name}/probs/instance_mask/{class_name}_output/{class_name}_instance_mask.png",
                "--output-dir",
                f"output/{dataset_name}/probs/out/{args.model_name}_{args.short_side}/{class_name}_tracking_{args.model_name}_{args.short_side}",
                "--short-side", args.short_side,
                "--max-context-length", args.max_context_length,
                "--topk", args.topk,
            ], check=True)
        except:
            pass

    # Merge predictions for this dataset
    merge_json(
        input_dir=f"output/{dataset_name}/probs/out/{args.model_name}_{args.short_side}/",
        model_name=args.model_name,
        short_side=args.short_side,
        frames_dir=str(dataset / "frames"),
        output_dir=str(dataset / "annotations"),
    )

    # Remove intermediate results
    shutil.rmtree(
        f"output/{dataset_name}/",
        ignore_errors=True,
    )

print("\nAll datasets processed successfully.")