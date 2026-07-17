"""Mask refinement: upsampling and CRF."""

import torch
import torch.nn.functional as F


def upsample_mask(mask: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Upsample a feature-resolution binary mask to (H, W)."""
    return (
        F.interpolate(
            mask[None, None].float(),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        > 0.5
    )


@torch.no_grad()
def crf_refine(
    crf_module,
    band_px: int,
    p_core: float,
    tgt_image: torch.Tensor,
    init_mask: torch.Tensor,
) -> torch.Tensor:
    """Refine a binary mask using GPU CRF on a narrow band around the boundary."""
    from utils.data import denormalize

    device = tgt_image.device
    H, W = init_mask.shape

    r = max(1, band_px // 2)
    ys, xs = torch.meshgrid(
        torch.arange(-r, r + 1, device=device),
        torch.arange(-r, r + 1, device=device),
        indexing="ij",
    )
    kernel = ((xs**2 + ys**2) <= r * r).float()[None, None]
    m0 = init_mask.float()[None, None]
    y = F.conv2d(m0, kernel, padding=r)
    ksum = float(kernel.sum().item())
    dil = (y > 0).squeeze()
    ero = (y == ksum).squeeze()
    band = dil ^ ero
    inside_core = ero
    outside_core = ~dil

    prob_fg = init_mask.float().clone()
    prob_fg[inside_core] = p_core
    prob_fg[outside_core] = 1.0 - p_core
    prob_fg[band] = 0.5

    logit_fg = torch.logit(prob_fg.clamp(1e-6, 1 - 1e-6))
    logits = torch.stack([-logit_fg, logit_fg], dim=0).unsqueeze(0).contiguous()

    img01 = denormalize(tgt_image[0]).clamp(0, 1).unsqueeze(0)
    logits_ref = crf_module(img01.to(device=device, dtype=torch.float32), logits)

    mask_crf = logits_ref[0, 1] > logits_ref[0, 0]
    result = init_mask.clone()
    result[band] = mask_crf[band]
    result[inside_core] = True
    result[outside_core] = False
    return result
