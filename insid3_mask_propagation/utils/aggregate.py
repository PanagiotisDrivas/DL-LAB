import numpy as np


def combine_masks(class_masks, classes):

    if len(class_masks) == 0:
        raise RuntimeError("No masks available")

    first = next(iter(class_masks.values()))

    h, w = first.shape

    semantic = np.zeros((h, w), dtype=np.uint8)

    for idx, cls in enumerate(classes, start=1):

        mask = class_masks.get(cls)

        if mask is None:
            continue

        semantic[mask > 0] = idx

    return semantic
