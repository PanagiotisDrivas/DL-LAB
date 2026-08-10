# # Copied from: https://github.com/facebookresearch/dinov3/blob/main/hubconf.py

# import torch

# # _DINOV3_BASE_URL = "C:/Users/ghost/.cache/torch/hub/checkpoints/"
# _DINOV3_BASE_URL = "/home/pandri2013/DL-LAB/SPINO"

# def _make_dinov3_model_name(arch_name: str, patch_size: int) -> str:
#     compact_arch_name = arch_name.replace("_", "")[:4]
#     return f"dinov3_{compact_arch_name}{patch_size}"


# def _make_dinov3_model(
#         *,
#         arch_name: str = "vit_large",
#         img_size: int = 518,
#         patch_size: int = 16,
#         init_values: float = 1.0,
#         ffn_layer: str = "mlp",
#         block_chunks: int = 0,
#         pretrained: bool = True,
#         **kwargs,
# ):
#     from models.vit import vision_transformer as vits

#     model_name = _make_dinov3_model_name(arch_name, patch_size)
#     vit_kwargs = dict(
#         img_size=img_size,
#         patch_size=patch_size,
#         init_values=init_values,
#         ffn_layer=ffn_layer,
#         block_chunks=block_chunks,
#     )
#     vit_kwargs.update(**kwargs)
#     model = vits.__dict__[arch_name](**vit_kwargs)
  
#     if pretrained:
#         print(model_name)
#         url = _DINOV3_BASE_URL + f"/{model_name}_pretrain.pth"
#         state_dict = torch.hub.load
#         state_dict = torch.load(url, map_location="cpu")
#         model.load_state_dict(state_dict, strict=False)
#         result = model.load_state_dict(state_dict, strict=False)
#         print("MISSING:", len(result.missing_keys), result.missing_keys[:10])
#         print("UNEXPECTED:", len(result.unexpected_keys), result.unexpected_keys[:10])
#     return model


# def dinov3_vits16(*, pretrained: bool = True, **kwargs):
#     """
#     DINOv3 ViT-S/16 model (optionally) pretrained on the LVD-162M dataset.
#     """
#     return _make_dinov3_model(arch_name="vit_small", pretrained=pretrained, **kwargs)


# def dinov3_vitb16(*, pretrained: bool = True, **kwargs):
#     """
#     DINOv3 ViT-B/16 model pretrained on the LVD-162M dataset.
#     """
#     return _make_dinov3_model(arch_name="vit_base", pretrained=pretrained, **kwargs)


# def dinov3_vitl16(*, pretrained: bool = True, **kwargs):
#     """
#     DINOv3 ViT-L/16 model (optionally) pretrained on the LVD-162M dataset.
#     """
#     return _make_dinov3_model(arch_name="vit_large", pretrained=pretrained, **kwargs)


# def dinov3_vitg16(*, pretrained: bool = True, **kwargs):
#     """
#     DINOv3 ViT-g/16 model (optionally) pretrained on the LVD-142M dataset.
#     """
#     return _make_dinov3_model(arch_name="vit_giant2", ffn_layer="swiglufused",
#                               pretrained=pretrained, **kwargs)


# --------------------------------------------------------------------------------------------------------------------------------------------------------


import sys
# import torch
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent

DINOV3_REPO_DIR = str(_THIS_DIR.parent / "dinov3_dict")
if DINOV3_REPO_DIR not in sys.path:
    sys.path.insert(0, DINOV3_REPO_DIR)

from dinov3.hub.backbones import (
    dinov3_vits16 as _dinov3_vits16,
    dinov3_vitb16 as _dinov3_vitb16,
    dinov3_vitl16 as _dinov3_vitl16,
    dinov3_vit7b16 as _dinov3_vit7b16,
)

CHECKPOINTS_DIR = _THIS_DIR.parent.parent.parent / "checkpoints"

DINOV3_WEIGHTS = {
    "vits16": str(CHECKPOINTS_DIR / "dinov3_vits16.pth"),
    "vitb16": str(CHECKPOINTS_DIR / "dinov3_vitb16.pth"),
    "vitl16": str(CHECKPOINTS_DIR / "dinov3_vitl16.pth"),
    "vit7b16": str(CHECKPOINTS_DIR / "dinov3_vit7b16.pth"),
}


def dinov3_vits16(pretrained: bool = True, **kwargs):
    return _dinov3_vits16(pretrained=pretrained, weights=DINOV3_WEIGHTS["vits16"] if pretrained else None)


def dinov3_vitb16(pretrained: bool = True, **kwargs):
    return _dinov3_vitb16(pretrained=pretrained, weights=DINOV3_WEIGHTS["vitb16"] if pretrained else None)


def dinov3_vitl16(pretrained: bool = True, **kwargs):
    return _dinov3_vitl16(pretrained=pretrained, weights=DINOV3_WEIGHTS["vitl16"] if pretrained else None)


def dinov3_vitg16(pretrained: bool = True, **kwargs):
    # DINOv3 has no "giant" size — largest available is vit7b16
    return _dinov3_vit7b16(pretrained=pretrained, weights=DINOV3_WEIGHTS["vit7b16"] if pretrained else None)
