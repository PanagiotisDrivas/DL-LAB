# Trimmed hubconf.py - backbone entrypoints only (dinov3_vits16, vitb16, vitl16, etc.)
# Original also exposed classifiers/depthers/detectors/dinotxt/segmentors entrypoints,
# which pull in unrelated detection/depth/text-tower code not needed for feature extraction.

from dinov3.hub.backbones import (
    dinov3_convnext_base,
    dinov3_convnext_large,
    dinov3_convnext_small,
    dinov3_convnext_tiny,
    dinov3_vit7b16,
    dinov3_vitb16,
    dinov3_vith16plus,
    dinov3_vitl16,
    dinov3_vitl16plus,
    dinov3_vits16,
    dinov3_vits16plus,
)

dependencies = ["torch", "numpy"]
