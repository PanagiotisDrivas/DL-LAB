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

1. **Extract:** Gather consecutive frames and the corresponding ground-truth instance mask (`gtFine`) for the **first frame** of a Cityscapes city sequence.
2. **Structure:** Format the data into the `temporal_propagation/input/` directory.
3. **Propagate:** Run the tracking scripts from within the `temporal_propagation/` directory to generate tracking sequences.

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
        │   ├── frames/
        │   │   ├── 000000.png
        │   │   ├── 000001.png
        │   │   └── ...
        │   └── first_mask/
        │       └── annotations.json
        └── frankfurt/
            ├── frames/
            └── first_mask/
                └── annotations.json

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

Place all consecutive sequence images here. The pipeline uses the lowest index frame (e.g., `000000.png`) as the tracking anchor.

```text
temporal_propagation/input/<dataset_name>/frames/
├── 000000.png
├── 000001.png
└── 000002.png

```

### Reference Frame Annotation

The instance mask for the very first frame must be formatted as a JSON file and placed here:

```text
temporal_propagation/input/<dataset_name>/first_mask/annotations.json

```

> **Note on Cityscapes Conversion:** To populate these folders, collect sequence frames from `leftImg8bit_sequence/` and convert the initial frame's corresponding mask from `gtFine/` into the target `annotations.json` file.

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

---

## 6. Output Structure

Once processing completes, tracking outputs are saved under the `temporal_propagation/output/` directory, and finalized merged annotations are copied directly back into your dataset folder:

```text
temporal_propagation/input/<dataset_name>/annotations/   # Final merged tracking annotations
temporal_propagation/output/<dataset_name>/              # Frame-by-frame predicted mask visualizations
Since frame by frame is not needed we delete it but you can view it by commenting the deletion comment in infer.py and infer_all.py

```