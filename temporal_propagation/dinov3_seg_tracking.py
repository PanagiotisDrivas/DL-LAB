#!/usr/bin/env python
"""
DINOv3 Segmentation Tracking Pipeline
======================================

Modular version of Meta's segmentation_tracking.ipynb notebook.
Given a folder of video frames (.jpg/.png) and a single GT instance mask
for the first frame, propagates the mask to all frames via DINOv3 feature
similarity, then exports frame/mask pairs to an output folder.

USAGE
-----
python dinov3_seg_tracking.py \
    --dinov3-location /Users/sakila/Desktop/dinov3 \
    --weights /Users/sakila/Desktop/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    --model-name dinov3_vits16 \
    --frames-dir /path/to/frames \
    --first-mask /path/to/first_frame_mask.png \
    --output-dir /path/to/output

INPUT FOLDER FORMAT
--------------------
frames-dir/
    000001.jpg
    000002.jpg
    ...
first-mask: a single .png where pixel value 0 = background,
            1..N = instance/class indices (uint8), matching frame 000001.

OUTPUT FOLDER FORMAT
---------------------
output-dir/
    frames/NNNNNN.png       - original frame, full resolution
    masks/NNNNNN.png        - predicted class-index mask, full resolution (the actual usable label)
    masks/NNNNNN_color.png  - colorized mask, for visual inspection only
    overlays/NNNNNN.png     - frame + colorized mask blended, for visual QA
"""

from __future__ import annotations

import argparse
import datetime
import functools
import gc
import logging
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import json
import torch.nn.functional as F
import torchvision.transforms as TVT
import torchvision.transforms.functional as TVTF
from PIL import Image
from torch import Tensor, nn
from tqdm import tqdm
from instance_segmentation.class_definitions import ID_TO_CLASS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("dinov3_seg_tracking")

# Critical: this is pure inference, no training. Without this, autograd tracks every
# op in the loop below and the computation graph grows unbounded across iterations
# (features_queue/probs_queue hold references that keep it alive) -> memory blows up
# and iterations get much slower.
torch.set_grad_enabled(False)


# --------------------------------------------------------------------------
# Device
# --------------------------------------------------------------------------

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def print_memory(device: str, label: str = "") -> None:
    prefix = f"[{label}] " if label else ""
    if device == "cuda":
        log.info(f"{prefix}Peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    elif device == "mps":
        log.info(f"{prefix}Peak MPS memory: {torch.mps.current_allocated_memory() / 2**30:.2f} GB")
    else:
        log.info(f"{prefix}Running on CPU, no memory stats available")


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------

def load_model(dinov3_location: str, model_name: str, weights_path: str, device: str) -> tuple[nn.Module, int, int]:
    dinov3_location = str(dinov3_location)
    weights_path = Path(weights_path)

    assert (Path(dinov3_location) / "hubconf.py").exists(), (
        f"dinov3_location doesn't point to a valid clone: {dinov3_location}"
    )
    assert weights_path.exists(), f"Weights file missing: {weights_path}"

    model = torch.hub.load(
        repo_or_dir=dinov3_location,
        model=model_name,
        source="local",
        weights=str(weights_path),
    )
    model.to(device)
    model.eval()

    patch_size = model.patch_size
    embed_dim = model.embed_dim
    log.info(f"Loaded {model_name}: patch_size={patch_size}, embed_dim={embed_dim}")
    print_memory(device, "after model load")
    return model, patch_size, embed_dim


@torch.compile(disable=True)
def forward(model: nn.Module, img: Tensor) -> Tensor:
    """img: [3, H, W] already normalized. Returns L2-normalized patch features [h, w, D]."""
    feats = model.get_intermediate_layers(img.unsqueeze(0), n=1, reshape=True)[0]  # [1, D, h, w]
    feats = feats.movedim(-3, -1)  # [1, h, w, D]
    feats = F.normalize(feats, dim=-1, p=2)
    return feats.squeeze(0)  # [h, w, D]


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_frames(frames_dir: str) -> list[Image.Image]:
    frames_dir = Path(frames_dir)
    paths = sorted(
        [p for p in frames_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    )
    assert len(paths) > 0, f"No frames found in {frames_dir}"
    frames = [Image.open(p).convert("RGB") for p in paths]
    log.info(f"Loaded {len(frames)} frames from {frames_dir}")
    return frames


def load_first_mask(first_mask_path: str):

    mask = np.array(Image.open(first_mask_path))

    if mask.ndim == 3:
        raise ValueError("Expected single-channel instance mask.")

    ids = np.unique(mask)

    object_ids = ids[ids != 0]

    id_mapping = {}

    remapped = np.zeros_like(mask, dtype=np.uint16)

    for new_id, old_id in enumerate(object_ids, start=1):

        remapped[mask == old_id] = new_id

        # recover class from Cityscapes encoding
        class_id = old_id // 1000

        id_mapping[new_id] = {
            "instanceId": int(old_id),
            "classId": int(class_id),
        }

    num_masks = len(object_ids) + 1

    log.info(f"Original IDs : {object_ids}")
    log.info(f"Tracking IDs : {np.unique(remapped)}")

    return remapped, num_masks, id_mapping


def mask_to_rgb(mask: np.ndarray | Tensor, num_masks: int) -> np.ndarray:
    if isinstance(mask, Tensor):
        mask = mask.cpu().numpy()
    background = mask == 0
    mask = mask - 1
    n = num_masks - 1
    if n <= 10:
        mask_rgb = plt.get_cmap("tab10")(mask)[..., :3]
    elif n <= 20:
        mask_rgb = plt.get_cmap("tab20")(mask)[..., :3]
    else:
        mask_rgb = plt.get_cmap("gist_rainbow")(mask / max(n - 1, 1))[..., :3]
    mask_rgb = (mask_rgb * 255).astype(np.uint8)
    mask_rgb[background, :] = 0
    return mask_rgb


# --------------------------------------------------------------------------
# Resizing
# --------------------------------------------------------------------------

class ResizeToMultiple(nn.Module):
    def __init__(self, short_side: int, multiple: int):
        super().__init__()
        self.short_side = short_side
        self.multiple = multiple

    def _round_up(self, side: float) -> int:
        return math.ceil(side / self.multiple) * self.multiple

    def forward(self, img):
        old_width, old_height = TVTF.get_image_size(img)
        if old_width > old_height:
            new_height = self._round_up(self.short_side)
            new_width = self._round_up(old_width * new_height / old_height)
        else:
            new_width = self._round_up(self.short_side)
            new_height = self._round_up(old_height * new_width / old_width)
        return TVTF.resize(img, [new_height, new_width], interpolation=TVT.InterpolationMode.BICUBIC)


def build_transform(short_side: int, patch_size: int) -> TVT.Compose:
    return TVT.Compose(
        [
            ResizeToMultiple(short_side=short_side, multiple=patch_size),
            TVT.ToTensor(),
            TVT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


# --------------------------------------------------------------------------
# Core propagation algorithm
# --------------------------------------------------------------------------

@torch.compile(disable=True)
def propagate(
    current_features: Tensor,  # [h", w", D]
    context_features: Tensor,  # [t, h, w, D]
    context_probs: Tensor,  # [t, h, w, M]
    neighborhood_mask: Tensor,  # [h", w", h, w]
    topk: int,
    temperature: float,
) -> Tensor:
    t, h, w, M = context_probs.shape

    dot = torch.einsum("ijd, tuvd -> ijtuv", current_features, context_features)  # [h",w",t,h,w]
    dot = torch.where(neighborhood_mask[:, :, None, :, :], dot, -torch.inf)

    dot = dot.flatten(2, -1).flatten(0, 1)  # [h"w", thw]
    k_th_largest = torch.topk(dot, dim=1, k=topk).values
    dot = torch.where(dot >= k_th_largest[:, -1:], dot, -torch.inf)

    weights = F.softmax(dot / temperature, dim=1)
    current_probs = torch.mm(weights, context_probs.flatten(0, 2))  # [h"w", M]
    current_probs = current_probs / current_probs.sum(dim=1, keepdim=True)

    return current_probs.unflatten(0, (h, w))  # [h", w", M]


@functools.lru_cache()
def make_neighborhood_mask(h: int, w: int, size: float, shape: str, device: str) -> Tensor:
    ij = torch.stack(
        torch.meshgrid(
            torch.arange(h, dtype=torch.float32, device=device),
            torch.arange(w, dtype=torch.float32, device=device),
            indexing="ij",
        ),
        dim=-1,
    )
    ord = 2 if shape == "circle" else torch.inf if shape == "square" else None
    if ord is None:
        raise ValueError(f"Invalid shape={shape}")
    norm = torch.linalg.vector_norm(ij[:, :, None, None, :] - ij[None, None, :, :, :], ord=ord, dim=-1)
    return norm <= size


def postprocess_probs(probs: Tensor) -> Tensor:
    """probs: [B, M, H, W]"""
    vmin = probs.flatten(2, 3).min(dim=2).values
    vmax = probs.flatten(2, 3).max(dim=2).values
    probs = (probs - vmin[:, :, None, None]) / (vmax[:, :, None, None] - vmin[:, :, None, None])
    return torch.nan_to_num(probs, nan=0)


# --------------------------------------------------------------------------
# Main video processing loop
# --------------------------------------------------------------------------

@torch.no_grad()
def process_video(
    model: nn.Module,
    frames: list[Image.Image],
    first_mask_np: np.ndarray,
    num_masks: int,
    device: str,
    patch_size: int,
    short_side: int = 336,
    max_context_length: int = 3,
    neighborhood_size: float = 12,
    neighborhood_shape: str = "circle",
    topk: int = 5,
    temperature: float = 0.2,
) -> tuple[Tensor, Tensor]:
    """Returns (mask_predictions [T,H,W] uint8, mask_probabilities [T,M,H,W] float32)."""
    num_frames = len(frames)
    mask_height, mask_width = first_mask_np.shape

    transform = build_transform(short_side, patch_size)

    first_frame = transform(frames[0]).to(device)
    _, frame_height, frame_width = first_frame.shape
    feats_height, feats_width = frame_height // patch_size, frame_width // patch_size
    log.info(f"Feature map size (h, w): {feats_height}, {feats_width}")

    first_mask = torch.from_numpy(first_mask_np).to(device, dtype=torch.long)
    first_mask = F.interpolate(
        first_mask[None, None, :, :].float(), (feats_height, feats_width), mode="nearest-exact"
    )[0, 0].long()
    first_probs = F.one_hot(first_mask, num_masks).float()

    first_feats = forward(model, first_frame)

    neighborhood_mask = make_neighborhood_mask(
        feats_height, feats_width, size=neighborhood_size, shape=neighborhood_shape, device=device
    )

    mask_predictions = torch.zeros([num_frames, mask_height, mask_width], dtype=torch.uint8)
    mask_predictions[0, :, :] = torch.from_numpy(first_mask_np)

    mask_probabilities = torch.zeros([num_frames, num_masks, mask_height, mask_width])
    mask_probabilities[0, :, :, :] = F.one_hot(
        torch.from_numpy(first_mask_np).long(), num_masks
    ).movedim(-1, -3)

    features_queue: list[Tensor] = []
    probs_queue: list[Tensor] = []

    start = time.perf_counter()
    for frame_idx in tqdm(range(1, num_frames), desc="Processing"):
        current_frame = transform(frames[frame_idx]).to(device)
        current_feats = forward(model, current_frame)

        context_feats = torch.stack([first_feats, *features_queue], dim=0)
        context_probs = torch.stack([first_probs, *probs_queue], dim=0)

        current_probs = propagate(
            current_feats, context_feats, context_probs, neighborhood_mask, topk, temperature
        )

        features_queue.append(current_feats)
        probs_queue.append(current_probs)
        if len(features_queue) > max_context_length:
            features_queue.pop(0)
        if len(probs_queue) > max_context_length:
            probs_queue.pop(0)

        # bilinear upsampling of continuous probabilities -> smoother mask boundaries
        current_probs = F.interpolate(
            current_probs.movedim(-1, -3)[None, :, :, :],
            size=(mask_height, mask_width),
            # mode="nearest",
            mode="bilinear",
            align_corners=True,
        )
        current_probs = postprocess_probs(current_probs).squeeze(0)
        mask_probabilities[frame_idx, :, :, :] = current_probs
        mask_predictions[frame_idx, :, :] = torch.argmax(current_probs, dim=0).to(dtype=torch.uint8)

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

    elapsed = time.perf_counter() - start
    log.info(f"Processing time: {datetime.timedelta(seconds=round(elapsed))}")
    log.info(f"Mask predictions: shape={mask_predictions.shape}, dtype={mask_predictions.dtype}")

    return mask_predictions, mask_probabilities


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------
def polygon_from_mask(binary_mask):
    """
    Convert a binary mask into a polygon.

    - Keeps only the largest connected component.
    - Uses CHAIN_APPROX_NONE to preserve contour points.
    - Applies a very light polygon simplification.
    """

    binary_mask = binary_mask.astype(np.uint8)

    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    if len(contours) == 0:
        return []

    # Keep only the largest contour
    contour = max(contours, key=cv2.contourArea)

    # Ignore tiny objects
    if cv2.contourArea(contour) < 3:
        return []

    # Optional: very light simplification
    epsilon = 0.0005 * cv2.arcLength(contour, True)
    contour = cv2.approxPolyDP(contour, epsilon, True)

    polygon = contour.reshape(-1, 2).tolist()

    return polygon


def save_json(
    mask,
    id_mapping,
    output_json,
):
    H, W = mask.shape

    objects = []

    ids = np.unique(mask)

    # remove background
    ids = ids[ids != 0]

    for tracking_id in ids:

        binary = mask == tracking_id

        polygons = polygon_from_mask(binary)

        if len(polygons) == 0:
            continue

        info = id_mapping[int(tracking_id)]

        class_id = info["classId"]

        objects.append(
            {
                # "instanceId": info["instanceId"],
                # "classId": class_id,
                "label": ID_TO_CLASS.get(class_id, "unknown"),
                # "area": int(binary.sum()),
                "polygon": polygons,
            }
        )

    data = {
        "imgHeight": int(H),
        "imgWidth": int(W),
        "objects": objects,
    }

    with open(output_json, "w") as f:
        json.dump(
            data,
            f,
            indent=2,
        )

def export_results(
    frames: list[Image.Image],
    mask_predictions: Tensor,
    num_masks: int,
    output_dir: str,
    id_mapping: dict,
    save_color: bool = True,
    save_overlay: bool = True,
    overlay_alpha: float = 0.5,
) -> None:

    output_dir = Path(output_dir)

    frames_dir = output_dir / "frames"
    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "overlays"
    json_dir = output_dir / "json"

    frames_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    if save_overlay:
        overlays_dir.mkdir(parents=True, exist_ok=True)

    num_frames = len(frames)

    for i in tqdm(range(num_frames), desc="Exporting"):

        frame_img = frames[i].convert("RGB")

        frame_img.save(
            frames_dir / f"{i:06d}.png"
        )

        mask_np = mask_predictions[i].cpu().numpy()

        Image.fromarray(mask_np).save(
            masks_dir / f"{i:06d}.png"
        )

        # --------------------------
        # Save JSON annotation
        # --------------------------

        save_json(
            mask_np,
            id_mapping,
            json_dir / f"{i:06d}.json",
        )

        # --------------------------
        # Save color visualization
        # --------------------------

        if save_color or save_overlay:

            mask_rgb = mask_to_rgb(mask_np, num_masks)

            if save_color:

                Image.fromarray(mask_rgb).save(
                    masks_dir / f"{i:06d}_color.png"
                )

            if save_overlay:

                overlay = Image.blend(
                    frame_img,
                    Image.fromarray(mask_rgb),
                    alpha=overlay_alpha,
                )

                overlay.save(
                    overlays_dir / f"{i:06d}.png"
                )

    log.info(
        f"Exported {num_frames} frames, masks, overlays and JSON annotations to {output_dir}"
    )

# --------------------------------------------------------------------------
# End-to-end pipeline
# --------------------------------------------------------------------------

def run_pipeline(
    dinov3_location: str,
    weights: str,
    model_name: str,
    frames_dir: str,
    first_mask: str,
    output_dir: str,
    short_side: int = 336,
    max_context_length: int = 3,
    neighborhood_size: float = 12,
    neighborhood_shape: str = "circle",
    topk: int = 5,
    temperature: float = 0.2,
) -> None:
    device = get_device()
    log.info(f"Using device: {device}")

    model, patch_size, _ = load_model(dinov3_location, model_name, weights, device)
    frames = load_frames(frames_dir)
    first_mask_np, num_masks, id_mapping = load_first_mask(first_mask)

    assert frames[0].size[::-1] == first_mask_np.shape, (
        f"Frame size {frames[0].size[::-1]} doesn't match mask size {first_mask_np.shape} "
        "(first frame and first mask must be the same resolution)"
    )

    mask_predictions, mask_probabilities = process_video(
        model=model,
        frames=frames,
        first_mask_np=first_mask_np,
        num_masks=num_masks,
        device=device,
        patch_size=patch_size,
        short_side=short_side,
        max_context_length=max_context_length,
        neighborhood_size=neighborhood_size,
        neighborhood_shape=neighborhood_shape,
        topk=topk,
        temperature=temperature,
    )

    # free the large probability tensor before export; only mask_predictions is needed on disk
    del mask_probabilities
    gc.collect()

    export_results(frames, mask_predictions, num_masks, output_dir, id_mapping=id_mapping,)
    print_memory(device, "final")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DINOv3 segmentation tracking, folder-in / folder-out")
    p.add_argument("--dinov3-location", required=True, help="Path to local dinov3 repo clone")
    p.add_argument("--model-name", default="dinov3_vits16", help="e.g. dinov3_vits16, dinov3_vitl16")
    p.add_argument("--frames-dir", required=True, help="Folder of input video frames (.jpg/.png)")
    p.add_argument("--first-mask", required=True, help="Path to first-frame GT mask (.png)")
    p.add_argument("--output-dir", required=True, help="Where to write frames/masks/overlays")
    p.add_argument("--short-side", type=int, default=512, help="Forward resolution short side")
    p.add_argument("--max-context-length", type=int, default=1)
    p.add_argument("--neighborhood-size", type=float, default=12)
    p.add_argument("--neighborhood-shape", default="circle", choices=["circle", "square"])
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    
    if args.model_name=="dinov3_vits16":
        weights = "../checkpoints/dinov3_vits16.pth"
    if args.model_name=="dinov3_vitb16":
        weights = "../checkpoints/dinov3_vitb16.pth"
    
    run_pipeline(
        dinov3_location=args.dinov3_location,
        weights=weights,
        model_name=args.model_name,
        frames_dir=args.frames_dir,
        first_mask=args.first_mask,
        output_dir=output_dir,
        short_side=args.short_side,
        max_context_length=args.max_context_length,
        neighborhood_size=args.neighborhood_size,
        neighborhood_shape=args.neighborhood_shape,
        topk=args.topk,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()