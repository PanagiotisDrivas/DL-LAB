from typing import List, Optional
import torch.nn.functional as F
import pytorch_lightning as pl
import torch
from models_v3.dino_v3 import (
    dinov3_vits16, dinov3_vitb16,
    dinov3_vitl16, dinov3_vitg16
)

from models.dino_v2 import (
    dinov2_vitb14,
    dinov2_vitg14,
    dinov2_vitl14,
    dinov2_vits14,
)

from torch import nn


class FineTuner(pl.LightningModule):
    def __init__(self, dinov2_vit_model:str, dinov3_vit_model: str, dinov3:int, debug:int, blocks: Optional[List[int]] = None,
                 upsample_factor: Optional[float] = None):
        super().__init__()
        self.dinov3_vit_model = dinov3_vit_model
        self.dinov2_vit_model = dinov2_vit_model
        self.blocks = blocks
        self.upsample_factor = upsample_factor
        self.dinov3_flag = dinov3
        self.debug = debug
        if dinov3:
            if dinov3_vit_model == 'vits16':
                self.encoder = dinov3_vits16(pretrained=True)
            elif dinov3_vit_model == 'vitb16':
                self.encoder = dinov3_vitb16(pretrained=True)
            elif dinov3_vit_model == 'vitl16':
                self.encoder = dinov3_vitl16(pretrained=True)
            elif dinov3_vit_model == 'vitg16':
                self.encoder = dinov3_vitg16(pretrained=True)
            else:
                raise ValueError(f'Unknown model {dinov3_vit_model}')
        else:
            if dinov2_vit_model == 'vits14':
                self.encoder = dinov2_vits14(pretrained=True)
            elif dinov2_vit_model == 'vitb14':
                self.encoder = dinov2_vitb14(pretrained=True)
            elif dinov2_vit_model == 'vitl14':
                self.encoder = dinov2_vitl14(pretrained=True)
            elif dinov2_vit_model == 'vitg14':
                self.encoder = dinov2_vitg14(pretrained=True)
            else:
                raise ValueError(f'Unknown model {dinov2_vit_model}')

        self.feat_dim = self.encoder.num_features
        self.patch_size = self.encoder.patch_size
        self.encoder.mask_token = None  # can't use ddp_find_unused_parameters_false otherwise
        for param in self.encoder.parameters():  # freeze backbone
            param.requires_grad = False

        if blocks is None:
            self.num_blocks = 1
        else:
            self.num_blocks = len(blocks)

    def forward_encoder(self, img: torch.Tensor, feature_key: str = 'x'):
        if self.dinov3_flag:
            n = 4
        else:
            n = 1
        # n=1
        # n=1 extracts only the final block
        block_outputs = self.encoder.get_intermediate_layers(
            img, 
            n=n, 
            reshape=True, 
            return_class_token=False
        )
        
        if n==1:    
            x = block_outputs[0] # Exact shape: (B, feat_dim, H, W)
        else:
            x = torch.stack(block_outputs, dim=0).mean(dim=0)

        if self.upsample_factor is not None:
            x = nn.functional.interpolate(x, scale_factor=self.upsample_factor, mode='bilinear', align_corners=False)  

        if self.debug:
            import matplotlib.pyplot as plt
            activation = x[0]              # (384, 448, 896)
            heatmap = activation.mean(dim=0)

            # Normalize
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

            plt.imshow(heatmap.cpu(), cmap="jet")
            plt.axis("off")
            if self.dinov3_flag:
                plt.savefig(f"test/dinov3/activation_mean_{n}_{str(img.shape[-1])}_dinov3.png", bbox_inches="tight", pad_inches=0)
            else:
                plt.savefig(f"test/dinov2/activation_mean_{n}_{str(img.shape[-1])}.png", bbox_inches="tight", pad_inches=0)

            plt.close()
            exit()
            
        return x

    # def forward_encoder(self, img: torch.Tensor, feature_key: str = "x_norm_patchtokens"):
    #     """
    #     Multi-block DINOv3 feature extractor (dense map output).
    #     """

    #     img_h, img_w = img.shape[2:]
    #     patches_h = img_h // self.patch_size
    #     patches_w = img_w // self.patch_size

    #     return_attention_features = feature_key in ["q", "k", "v", "attn"]

    #     with torch.no_grad():
    #         block_outputs = self.encoder.forward_features(
    #             img,
    #             return_attention_features=return_attention_features,
    #             return_blocks=self.blocks
    #         )

    #         # if only final output
    #         if self.blocks is None:
    #             block_outputs = [block_outputs]

    #     outs = []

    #     for i, block in enumerate(block_outputs):

    #         # ----------------------------
    #         # DINOv3 SAFE FEATURE PICK
    #         # ----------------------------
    #         if feature_key not in block:
    #             raise KeyError(
    #                 f"Missing {feature_key} in block {i}. "
    #                 f"Available keys: {list(block.keys())}"
    #             )

    #         x = block[feature_key]  # (B, 1+P, C) or (B, P, C)

    #         # we only keep patch tokens
    #         if x.ndim == 3 and x.shape[1] > patches_h * patches_w:
    #             # remove CLS if present
    #             x = x[:, 1:, :]

    #         outs.append(x)

    #     # ---------------------------------------
    #     # concatenate across blocks (channel dim)
    #     # ---------------------------------------
    #     x = torch.cat(outs, dim=-1)  # (B, P, C * num_blocks)

    #     B, N, C = x.shape

    #     # ---------------------------------------
    #     # reshape to spatial map
    #     # ---------------------------------------
    #     x = x.transpose(1, 2).contiguous()  # (B, C, N)

    #     x = x.reshape(
    #         B,
    #         C,
    #         patches_h,
    #         patches_w
    #     )
    #     print(self.upsample_factor)
    #     # optional upsample
    #     if self.upsample_factor is not None:
    #         x = F.interpolate(
    #             x,
    #             scale_factor=self.upsample_factor,
    #             mode="bilinear",
    #             align_corners=False
    #         )
    #     import matplotlib.pyplot as plt
    #     activation = x[0]              # (384, 448, 896)
    #     heatmap = activation.mean(dim=0)

    #     # Normalize
    #     # heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    #     plt.imshow(heatmap.cpu(), cmap="jet")
    #     plt.axis("off")
    #     plt.savefig("test/activation_meanv3.png", bbox_inches="tight", pad_inches=0)
    #     plt.close()
    #     exit()
    #     return x