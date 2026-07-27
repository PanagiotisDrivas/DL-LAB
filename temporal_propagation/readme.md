Here is the updated README structure reflecting that the `input/` directory is located inside the `temporal_propagation/` folder.

The structure, paths, and commands have been updated to ensure accurate relative paths.

---

# Temporal Mask Propagation with DINOv3

This repository contains a pipeline to temporally propagate an initial instance mask across video frames using DINOv3 feature tracking.

---

## 1. Project Workflow

```
┌──────────────────────────┐     ┌──────────────────────────┐     ┌──────────────────────────┐
│  Cityscapes Source Data  │ ──> │   Processed Input Data   │ ──> │ DINOv3 Feature Tracking  │
│  (Frames + gtFine Mask)  │     │ (Inside temporal_prop/)  │     │   (Propagated Masks)     │
└──────────────────────────┘     └──────────────────────────┘     └──────────────────────────┘

```

1. **Extract:** Gather consecutive frames and the corresponding ground-truth instance mask (`gtFine`) for the **annotated (GT) frame** of a Cityscapes city sequence — by convention the 20th frame of a 30-frame snippet (19 frames before it, 10 after).
2. **Structure:** Format the data into the `temporal_propagation/input/` directory. The GT frame's own image is excluded from `frames/` and instead placed in `first_mask/` next to its annotation.
3. **Propagate:** Run the tracking scripts from within the `temporal_propagation/` directory. The GT mask is propagated **both forward** through the frames after it and **backward** (walking outward from the frame closest to the GT) through the frames before it, then stitched back into chronological order.

---

## 2. Repository Structure

Ensure your workspace is organized as follows:

```text
repository/
├── checkpoints/
│   ├── dinov3_vits16_pretrain.pth
│   └── dinov3_vitb16_pretrain.pth
│
└── temporal_propagation/
    ├── infer.py
    ├── infer_all.py
    ├── dinov3_seg_tracking.py
    └── input/                       <-- Input data resides here
        ├── bochum/
        │   ├── frames/                  <-- 29 frames: 19 before + 10 after the GT frame (GT frame itself excluded)
        │   │   ├── 000000.png
        │   │   ├── 000001.png
        │   │   └── ...
        │   └── first_mask/
        │       ├── annotations.json     <-- GT frame's polygon annotation
        │       └── <gt_frame>.png       <-- GT frame's own RGB image (needed to extract anchor features)
        └── frankfurt/
            ├── frames/
            └── first_mask/
                ├── annotations.json
                └── <gt_frame>.png

```

---

## 3. Prerequisites & Checkpoints

Download the required DINOv3 pre-trained weights and place them in the root `checkpoints/` directory. The pipeline expects these exact filenames:

* `checkpoints/dinov3_vits16.pth`
* `checkpoints/dinov3_vitb16.pth`

---

## 4. Input Data Preparation

The pipeline dynamically detects and processes city folders inside `temporal_propagation/input/`. Each city folder must contain:

### Video Frames

Place all consecutive sequence images here, **excluding** the annotated (GT) frame itself. By convention there are 19 frames before the GT frame and 10 after it (`--num-before-frames`, default `19`).

```text
temporal_propagation/input/<dataset_name>/frames/
├── 000000.png     <-- 19 frames before the GT frame
├── ...
├── 000018.png
├── 000019.png     <-- 10 frames after the GT frame
├── ...
└── 000028.png

```

### Reference Frame Annotation

The GT frame's instance mask (as a JSON file) **and** its own RGB image go in `first_mask/`. The image is required because the pipeline extracts DINOv3 features from it as the anchor for propagation in both directions:

```text
temporal_propagation/input/<dataset_name>/first_mask/
├── annotations.json
└── <gt_frame_name>.png

```

> **Note on Cityscapes Conversion:** To populate these folders, collect sequence frames from `leftImg8bit_sequence/`, set aside the 20th frame (the one matching `gtFine/`) into `first_mask/` along with its converted `annotations.json`, and place the remaining 29 frames in `frames/`.

The GT mask is propagated **forward** through the frames after it and **backward** through the frames before it (walking outward from the frame closest to the GT), then the two results are stitched back into chronological order — matching the original 29-frame ordering in `frames/`.

---

## 5. Running Temporal Propagation

Always execute the scripts from within the `temporal_propagation/` directory:

```bash
cd temporal_propagation

```

### Quick Start

To run with default settings (`dinov3_vitb16`, `topk=5`, context length `11`):

```bash
# Run a single city sequence
python infer.py --input-dir bochum

# Run all city sequences found in input/
python infer_all.py

```

### Advanced Usage (Single Dataset)

For custom tracking hyper-parameters, pass explicit arguments to `infer.py`:

```bash
python infer.py \
    --model-name dinov3_vitb16 \
    --input-dir bochum \
    --topk 5 \
    --max-context-length 11 \
    --short-side 768

```

### Configuration Arguments

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--model-name` | `str` | `dinov3_vitb16` | DINOv3 backbone architecture (`dinov3_vits16` or `dinov3_vitb16`). |
| `--input-dir` | `str` | *Required* | Name of the target dataset directory inside `input/` (e.g., `bochum`). |
| `--topk` | `int` | `5` | Number of nearest neighbor features used for tracking. |
| `--max-context-length` | `int` | `11` | Size of the temporal context window (number of reference frames). |
| `--short-side` | `int` | `768` | Rescales the short side of the input images to this resolution while maintaining aspect ratio. |
| `--num-before-frames` | `int` | `19` | How many of `frames/`'s frames come chronologically before the GT frame; the rest are propagated forward. |

---

## 6. Output Structure

Once processing completes, tracking outputs are saved under the `temporal_propagation/output/` directory, and finalized merged annotations are copied directly back into your dataset folder:

```text
temporal_propagation/input/<dataset_name>/annotations/   # Final merged tracking annotations
temporal_propagation/output/<dataset_name>/              # Frame-by-frame predicted mask visualizations
Since frame by frame is not needed we delete it but you can view it by commenting the deletion comment in infer.py and infer_all.py

```