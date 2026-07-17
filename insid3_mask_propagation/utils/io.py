from pathlib import Path


def load_references(reference_root, class_name):

    folder = reference_root / class_name

    images = sorted(folder.glob("*_img.png"))

    masks = sorted(folder.glob("*_mask.png"))

    return list(zip(images, masks))
