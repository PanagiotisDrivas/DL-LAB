# Copied from: https://github.com/facebookresearch/dinov3/blob/main/hubconf.py

import torch

_DINOV3_BASE_URL = "C:/Users/ghost/.cache/torch/hub/checkpoints/"


def _make_dinov3_model_name(arch_name: str, patch_size: int) -> str:
    compact_arch_name = arch_name.replace("_", "")[:4]
    return f"dinov3_{compact_arch_name}{patch_size}"


def _make_dinov3_model(
        *,
        arch_name: str = "vit_large",
        img_size: int = 518,
        patch_size: int = 16,
        init_values: float = 1.0,
        ffn_layer: str = "mlp",
        block_chunks: int = 0,
        pretrained: bool = True,
        **kwargs,
):
    from models.vit import vision_transformer as vits

    model_name = _make_dinov3_model_name(arch_name, patch_size)
    vit_kwargs = dict(
        img_size=img_size,
        patch_size=patch_size,
        init_values=init_values,
        ffn_layer=ffn_layer,
        block_chunks=block_chunks,
    )
    vit_kwargs.update(**kwargs)
    model = vits.__dict__[arch_name](**vit_kwargs)

    if pretrained:
        print(model_name)
        url = _DINOV3_BASE_URL + f"/{model_name}_pretrain.pth"
        state_dict = torch.hub.load
        state_dict = torch.load(url, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    return model


def dinov3_vits16(*, pretrained: bool = True, **kwargs):
    """
    DINOv3 ViT-S/16 model (optionally) pretrained on the LVD-162M dataset.
    """
    return _make_dinov3_model(arch_name="vit_small", pretrained=pretrained, **kwargs)


def dinov3_vitb16(*, pretrained: bool = True, **kwargs):
    """
    DINOv3 ViT-B/16 model pretrained on the LVD-162M dataset.
    """
    return _make_dinov3_model(arch_name="vit_base", pretrained=pretrained, **kwargs)


def dinov3_vitl16(*, pretrained: bool = True, **kwargs):
    """
    DINOv3 ViT-L/16 model (optionally) pretrained on the LVD-162M dataset.
    """
    return _make_dinov3_model(arch_name="vit_large", pretrained=pretrained, **kwargs)


def dinov3_vitg16(*, pretrained: bool = True, **kwargs):
    """
    DINOv3 ViT-g/16 model (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov3_model(arch_name="vit_giant2", ffn_layer="swiglufused",
                              pretrained=pretrained, **kwargs)
