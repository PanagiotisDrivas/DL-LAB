# Temporal Mask Propagation

This module temporally propagates a single ground-truth instance mask across a sequence of video frames using DINOv3 feature tracking, and writes the results directly into a SPINO-ready Cityscapes-style dataset.

---

# Directory Structure

```
temporal_propagation/

├── infer.py
├── infer_all.py
├── dinov3_seg_tracking.py
├── create_mask.py
├── config.py
│
├── instance_segmentation/
│   ├── class_definitions.py
│   └── create_instance_pipeline.py
│
├── utils/
│   └── merge.py
│
├── input/
│   ├── aachen/
│   │   └── first_mask/
│   │   └── frames/
│   ├── bremen1/
│   └── ...
│
└── output/

temporal_dataset/  
├── gtFine/
└── leftImg8bit_sequence/
```

---

# Pipeline Overview

The complete pipeline consists of three main steps:

1. **Build the Combined Instance Mask**
   - Rasterizes the GT frame's polygon annotation into a single instance mask spanning every class.

2. **Run DINOv3 Feature Tracking**
   - Propagates the combined mask outward from the GT frame, forward through later frames and backward through earlier frames, in one joint tracking run.

3. **Write the SPINO-Ready Annotations**
   - Renders `labelIds` / `instanceIds` / `color` PNGs for every propagated frame straight into `temporal_dataset/gtFine/<split>/<city>/`.

---

# Step 1: The `temporal_dataset` (Provided in google drive)

`temporal_dataset/` is a SPINO-ready Cityscapes-style dataset tree, with `leftImg8bit_sequence/` already populated with the sequence images..

---

# Step 2: Input Data Preparation (Provided in google drive)

The pipeline dynamically detects and processes sequence folders inside `/input/`. Each folder is one sequence (e.g. `bremen1`, `bremen2` for two different sequences of the same city) and must contain:

```
input/<sequence_name>/

├── frames/ 
│   ├── <city>_<seq>_000000_leftImg8bit.png
│   ├── ...
│   └── <city>_<seq>_000028_leftImg8bit.png
│
└── first_mask/
    ├── <city>_<seq>_<gt_frame>_gtFine_polygons.json  
    └── <city>_<seq>_<gt_frame>_leftImg8bit.png      
```

- **`frames/`** — all consecutive sequence images, **excluding** the annotated (GT) frame itself. By convention there are 19 frames before the GT frame and 10 after it (`--num-before-frames`, default `19`).
- **`first_mask/`** — the GT frame's polygon annotation *and* its own RGB image. The image is required because the pipeline extracts DINOv3 features from it as the anchor for propagation in both directions.

> **Note on multi-sequence cities:** Sequence names with a trailing digit (`bremen1`, `bremen2`, `darmstadt1/2/3`, …) are automatically merged into one shared `<city>/` folder under `temporal_dataset/gtFine/<split>/` — the trailing digit is stripped and does not need to be unique across sequences.

---

# Step 3: Running Temporal Propagation 

Always execute the scripts from within the `temporal_propagation/` directory:

```bash
cd temporal_propagation
```

### Quick Start

```bash
# Run a single sequence
python infer.py --input-dir bochum

# Run every sequence found in input/
python infer_all.py
```

### Advanced Usage (Single Sequence)

```bash
python infer.py \
    --model-name dinov3_vitb16 \
    --input-dir bochum \
    --topk 5 \
    --max-context-length 11 \
    --short-side 768 \
    --dataset-root ../temporal_dataset \
    --split train
```

### Common Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-name` | `dinov3_vitb16` | DINOv3 backbone (`dinov3_vits16`, `dinov3_vitb16`, `dinov3_vitl16`). |
| `--input-dir` | *required* | Sequence folder name inside `input/` (`infer.py` only — `infer_all.py` processes every folder). |
| `--topk` | `5` | Number of nearest-neighbor features used for tracking. |
| `--max-context-length` | `11` | Size of the temporal context window (number of reference frames). |
| `--short-side` | `768`/`1024` | Rescales the short side of input images to this resolution, keeping aspect ratio. |
| `--temperature` | `0.2` | Softmax temperature during propagation — lower is sharper/more decisive, higher blends more smoothly across matched classes. |
| `--num-before-frames` | `19` | How many of `frames/`'s frames come chronologically before the GT frame; the rest are propagated forward. |
| `--dataset-root` | `temporal_dataset` | Root of the SPINO-ready dataset tree that `labelIds`/`instanceIds`/`color` PNGs are written into. |
| `--split` | `train` | Split subfolder under `gtFine/` (e.g. `train`, `val`). |

---

# Example Commands

## Use the small DINOv3 backbone

```bash
python infer.py --input-dir bochum --model-name dinov3_vits16
```

## Widen the temporal context and neighbor search

```bash
python infer.py --input-dir bochum --max-context-length 15 --topk 8
```

## Write into the validation split

```bash
python infer.py --input-dir frankfurt --split val
```

## Process every sequence in `input/`

```bash
python infer_all.py --model-name dinov3_vitb16 --short-side 768
```

---

# Output

For each sequence processed, the pipeline:

1. Writes the joint DINOv3 tracking annotations back into `input/<sequence_name>/annotations/` (one `*_gtFine_polygons.json` per frame).
2. Renders `*_gtFine_labelIds.png`, `*_gtFine_instanceIds.png`, and `*_gtFine_color.png` for every frame directly into:

```
temporal_dataset/

└── gtFine/
    └── <split>/
        └── <city>/
            ├── <city>_<seq>_<frame>_gtFine_labelIds.png
            ├── <city>_<seq>_<frame>_gtFine_instanceIds.png
            └── <city>_<seq>_<frame>_gtFine_color.png
```

3. Deletes the intermediate `output/<sequence_name>/` directory (frame-by-frame tracking visualizations) once the annotations have been written. To inspect these instead of deleting them, comment out the cleanup step at the end of `infer.py` / `infer_all.py`.

---

# Complete Pipeline

```bash
cd temporal_propagation

# 1. Place temporal_dataset/ and input/ inside  temporal_propagation/, provided in Google Drive.

# 2. Propagate every sequence and write annotations into temporal_dataset/
python infer_all.py
```

Each sequence's propagated instance mask lands in `temporal_dataset/gtFine/<split>/<city>/`, ready for SPINO training against the already-provided `temporal_dataset/leftImg8bit_sequence/`.
