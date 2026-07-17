from pathlib import Path

### REFERENCE BANK
CITYSCAPES_ROOT = Path("../SPINO/panoptic_label_generator/cityscapes")
OUTPUT_REFRENCE_INSTANCE_BANK_ROOT = Path("reference_instance_bank")
OUTPUT_REFRENCE_SEMANTIC_BANK_ROOT = Path("reference_semantic_bank")

CLASSES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]

## CONFIG FOR REFERENCE BANK

CLASS_SET = set(CLASSES)
MAX_PER_CLASS = 11
MIN_AREA = 5000


# Inference
REFRENCE_INSTANCE_BANK_ROOT = Path("reference_instance_bank")
REFRENCE_SEMANTIC_BANK_ROOT = Path("reference_semantic_bank")
OUTPUT_ROOT = Path("aggregated_masks")
