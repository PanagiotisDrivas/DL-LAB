# INSID3

This module performs reference-based semantic segmentation using INSID3 and prepares the generated annotations for use with SPINO.

---

# Directory Structure

```
insid3_mask_propagation/

├── infer.py
├── generate_reference_semantic_mask.py
├── create_insid3_dataset_for_spino.py
├── create_mask.py
├── get_heads.py
├── config.py
├── create_label_mask.py
│
├── input/
│   ├── images/
│   ├── annotations/
│   └── masks/
│
├── reference_semantic_bank/
│
├── aggregated_masks/
│
└── insid3_dataset/
    ├── gtFine/
    └── leftImg8bit_sequence/
```

---

# Pipeline Overview

The complete pipeline consists of three main steps:

1. **Generate Semantic Reference Bank**
   - Creates reference images and semantic masks for each class.

2. **Run INSID3 Inference**
   - Uses the semantic reference bank to perform reference-based semantic segmentation.
   - Generates class-wise segmentation masks and polygon annotations.

3. **Prepare SPINO Dataset**
   - Converts INSID3 outputs into the required Cityscapes-style format for SPINO training.

---

# Step 1: Semantic Reference Bank

The pipeline uses a **Semantic Reference Bank** containing reference images and their corresponding semantic masks.

Each reference consists of:

- a reference image
- a semantic mask containing all pixels belonging to a specific semantic class present in that image

The reference bank paths and generation settings are configured in `config.py`.

Before generating the reference bank, update:

```python
CITYSCAPES_ROOT = Path("<path_to_cityscapes>")
```

to point to the Cityscapes dataset location.

The generated reference bank is stored according to:

```python
OUTPUT_REFERENCE_SEMANTIC_BANK_ROOT = Path("reference_semantic_bank")
```

The supported semantic classes are defined in:

```python
CLASSES = [
    "road",
    "sidewalk",
    "building",
    ...
]
```

Additional generation parameters can be controlled using:

```python
MAX_PER_CLASS = 11   # Maximum number of references per class
MIN_AREA = 5000      # Minimum object area
```

---

## Generating the Semantic Reference Bank

Run:

```bash
python generate_reference_semantic_mask.py
```

The generated references are stored as:

```
reference_semantic_bank/

├── car/
│   ├── car_000_img.png
│   ├── car_000_mask.png
│   ├── car_001_img.png
│   ├── car_001_mask.png
│   └── ...
│
├── person/
│
└── building/
```

Each class directory contains:

- `*_img.png` — reference image
- `*_mask.png` — semantic mask for the corresponding class

The generated masks contain all pixels belonging to the target semantic class.

---

# Step 2: Running INSID3 Inference

The script `infer.py` performs the complete INSID3 inference pipeline.

It:

1. Loads the semantic reference bank.
2. Performs reference-based semantic matching using INSID3.
3. Generates class-wise segmentation masks.
4. Aggregates the masks.
5. Creates polygon annotations from the final semantic masks.

---

## Input Images

Place images to be processed in:

```
input/images/
```

Example:

```
input/

├── images/
│   ├── image1.png
│   ├── image2.png
│   └── image3.png
│
├── annotations/
└── masks/
```

---

## Batch Inference

Process all images inside `input/images/`:

```bash
python infer.py
```

or specify another directory:

```bash
python infer.py --image-dir input/images
```

Images with an existing JSON annotation in `input/annotations/` are automatically skipped.

---

## Single Image Inference

Run inference on a single image:

```bash
python infer.py --image input/images/image1.png
```

---

# Common Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_size` | `base` | INSID3 model size (`small`, `base`, `large`) |
| `--max_instances` | `5` | Maximum number of reference images used per class |
| `--tau` | `0.7` | Clustering Threshold (0.7) Best for Cityscapes |
| `--threshold` | `0.2` | Mask merge threshold |
| `--object_dino` | Disabled | Enable DINOv3 attention-based head selection |
| `--remove_masks` | Disabled | Delete intermediate masks after JSON generation |

---

We also explored on OBJECT-DINO to retrieve object centric head and use the Value matrices for INSID3.
OBJECT-DINO : https://samyakr99.github.io/Object_dino/

# Example Commands

## Use the large INSID3 model

```bash
python infer.py --model_size large
```

## Use more reference images per class

```bash
python infer.py --max_instances 10
```

## Adjust similarity and merge thresholds

```bash
python infer.py --tau 0.7 --threshold 0.3
```

## Enable ObjectDINO

```bash
python infer.py --object_dino
```

## Remove intermediate masks after generating JSON annotations

```bash
python infer.py --remove_masks
```

---

# INSID3 Output

For each processed image, intermediate class masks are stored in:

```
aggregated_masks/

└── image1_base/
    ├── building.png
    ├── car.png
    ├── person.png
    ├── ...
    └── semantic_mask.png
```

The final INSID3 polygon annotations are generated in:

```
input/

└── annotations/
    └── image1.json
```

If `--remove-masks` is enabled, the corresponding directory inside `aggregated_masks/` is deleted after generating the JSON annotation.

---

# Step 3: Preparing Dataset for SPINO

After INSID3 inference, the generated annotations need to be converted into the format required by SPINO.

This is done using:

```bash
python create_insid3_dataset_for_spino.py
```

The script performs the following steps:

1. Renames annotation files according to the Cityscapes naming convention.
2. Generates Cityscapes-compatible semantic and instance masks.
3. Creates the required SPINO dataset directory structure.
4. Copies only images with corresponding annotations.

---

## Preparing Annotations

Before conversion, annotations are renamed from:

```
<city>_<sequence>_<frame>.json
```

to:

```
<city>_<sequence>_<frame>_gtFine_polygons.json
```

This matches the Cityscapes annotation format expected by SPINO.

---

## SPINO Dataset Structure

The generated dataset is stored as:

```
insid3_dataset/

├── gtFine/
│   └── train/
│       └── <city>/
│           ├── *_gtFine_polygons.json
│           ├── *_gtFine_labelIds.png
│           └── *_gtFine_instanceIds.png
│
└── leftImg8bit_sequence/
    └── train/
        └── <city>/
            └── *_leftImg8bit.png
```

Only images with corresponding INSID3 annotations are copied.

---

# Complete Pipeline

The complete INSID3-to-SPINO pipeline can be executed in the following order:

---

## Step 1: Generate Semantic Reference Bank

```bash
python generate_reference_semantic_mask.py
```

Creates reference images and semantic masks for each semantic class.

---

## Step 2: Run INSID3 Inference

```bash
python infer.py
```

Generates semantic masks and polygon annotations using INSID3.

---

## Step 3: Prepare SPINO Dataset

```bash
python create_insid3_dataset_for_spino.py
```

Converts INSID3 outputs into the required SPINO training format.

---