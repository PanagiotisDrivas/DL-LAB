import numpy as np
import torch
import cv2


def tensor_to_numpy(mask):

    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    return np.squeeze(mask).astype(np.uint8)


def normalize_mask(mask):

    return (mask > 0).astype(np.uint8)


def save_mask(mask, path):
    cv2.imwrite(str(path), mask * 255)
