from pathlib import Path
import argparse
import cv2
import shutil

from INSID3.models import build_insid3

from config import *
import random

from utils.io import load_references
from utils.infer import generate_class_mask
from utils.aggregate import combine_masks
from utils.mask import save_mask
from utils.mask_to_json import masks_to_json
from get_heads import dinov3_attention_clustering
from PIL import Image

def run(target_image, model_size, image_dir, max_instances=None, remove_masks=False, reference_mask_type="semantic", tau=0.6, threshold=0.2, object_dino=False):

    # --------------------------------
    # Output directory
    # --------------------------------


    if object_dino:
        img_objdino = Image.open(
                target_image
            ).convert("RGB")

        result = dinov3_attention_clustering(img_objdino)

        layers_and_heads = result["selections"]
        print(layers_and_heads)
    else:
        layers_and_heads = []
    

    image_name = Path(target_image).stem

    output_dir = Path("aggregated_masks") / f"{image_name}_{model_size}"

    # annotation directory
    image_dir = Path(image_dir).parent

    annotation_dir = image_dir / "annotations"

    annotation_dir.mkdir(parents=True, exist_ok=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Target image :", target_image)
    print("Model size   :", model_size)
    print("Output dir   :", output_dir)

    # --------------------------------
    # Build INSID3 model
    # --------------------------------

    model = build_insid3(model_size=model_size, image_size=1024, tau=tau, merge_threshold=threshold) #CHANGE SOMEE CONFIGGGSSSS HEERRREEEEEE


    class_masks = {}

    # --------------------------------
    # Class-wise inference
    # --------------------------------

    for cls in CLASSES:

        print(f"\nProcessing class: {cls}")

        if reference_mask_type=="semantic":
            references = load_references(REFERENCE_SEMANTIC_BANK_ROOT, cls)
        else:
            references = load_references(REFERENCE_INSTANCE_BANK_ROOT, cls)

        # Select number of reference images
        if max_instances is not None:

            references = references[:max_instances]

        print("References:", len(references))

        if len(references) == 0:
            continue

        mask = generate_class_mask(model, references, target_image)

        if mask is None:
            continue

        print("Mask pixels:", mask.sum())

        if mask.sum() == 0:
            continue

        class_masks[cls] = mask

        save_mask(mask, output_dir / f"{cls}.png")

    # --------------------------------
    # Combine semantic mask
    # --------------------------------

    if len(class_masks) == 0:

        print("No masks generated")

        return

    semantic_mask = combine_masks(class_masks, CLASSES)

    cv2.imwrite(str(output_dir / "semantic_mask.png"), semantic_mask)

    # --------------------------------
    # Generate polygon JSON
    # --------------------------------
    json_path = annotation_dir / f"{image_name}.json"


    masks_to_json(mask_root=output_dir, output_json=json_path)

    print("Saved annotation:", json_path)

    # --------------------------------
    # Remove intermediate masks
    # --------------------------------

    if remove_masks:

        shutil.rmtree(output_dir, ignore_errors=True)

        print("Removed:", output_dir)

    print("\nFinished:", image_name)


def process_all_images(image_dir, model_size, max_instances, remove_masks, reference_mask_type="semantic", tau=0.6, threshold=0.2, object_dino=False):

    image_dir = Path(image_dir)
    
    annotation_dir = image_dir.parent / "annotations"
    completed = {p.stem for p in annotation_dir.glob("*.json")}
    images = sorted(
        [
            p
            for p in image_dir.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
            and p.stem not in completed
        ]
    )
    random.shuffle(images)
    print(f"Found {len(images)} images")

    for image_path in images:

        print("\n" + "=" * 70)

        print(f"Processing image: {image_path.name}")

        print("=" * 70)

        try:

            run(
                target_image=str(image_path),
                model_size=model_size,
                image_dir=image_dir,
                max_instances=max_instances,
                remove_masks=remove_masks,
                reference_mask_type=reference_mask_type,
                tau=tau,
                threshold=threshold,
                object_dino=object_dino
            )

        except Exception as e:

            print(f"ERROR processing {image_path.name}")

            print(e)

    print("\nAll images completed")


def parse_args():

    parser = argparse.ArgumentParser(description="INSID3 reference-based segmentation")

    parser.add_argument(
        "--image_dir",
        type=str,
        default="input/images",
        help="Directory containing input images",
    )

    parser.add_argument(
        "--image", type=str, default=None, help="Run inference on a single image"
    )

    parser.add_argument(
        "--model_size",
        type=str,
        default="base",
        choices=["small", "base", "large"],
        help="INSID3 model size",
    )

    parser.add_argument(
        "--type",
        type=str,
        default="semantic",
        choices=["semantic"],
        help="type of reference bank",
    )

    parser.add_argument(
        "--max_instances",
        type=int,
        default=5,
        help="Maximum number of reference images per class",
    )

    parser.add_argument(
        "--remove_masks",
        action="store_true",
        default=True,
        help="Remove aggregated mask folders after JSON generation",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.7,
        help="TAU",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="threshold",
    )

    parser.add_argument(
        "--object_dino",
        action="store_true",
        default=False,
        help="Enable object DINO mode",
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    # --------------------------------
    # Single image mode
    # --------------------------------

    if args.image is not None:

        image_path = Path(args.image)

        if not image_path.exists():

            raise FileNotFoundError(f"Image not found: {image_path}")

        run(
            target_image=str(image_path),
            model_size=args.model_size,
            image_dir=image_path.parent,
            max_instances=args.max_instances,
            remove_masks=args.remove_masks,
            reference_mask_type=args.type,
            tau=args.tau,
            threshold=args.threshold,
            object_dino=args.object_dino,
        )

    # --------------------------------
    # Batch mode
    # --------------------------------

    else:

        process_all_images(
            image_dir=args.image_dir,
            model_size=args.model_size,
            max_instances=args.max_instances,
            remove_masks=args.remove_masks,
            reference_mask_type=args.type,
            tau=args.tau,
            threshold=args.threshold,
            object_dino=args.object_dino,
        )

