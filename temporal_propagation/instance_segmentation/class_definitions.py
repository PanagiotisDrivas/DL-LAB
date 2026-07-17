"""
Cityscapes semantic class definitions.

Every semantic class has

- semantic class id
- RGB color used for visualization

The colors follow the official Cityscapes palette.
"""

from __future__ import annotations

# ------------------------------------------------------------
# Semantic IDs
# ------------------------------------------------------------

CLASS_IDS = {
    # "unlabeled": 0,

    "road": 1,
    "sidewalk": 2,
    "building": 3,
    "wall": 4,
    "fence": 5,
    "pole": 6,
    "traffic light": 7,
    "traffic sign": 8,

    "vegetation": 9,
    "terrain": 10,
    "sky": 11,

    "person": 12,
    "rider": 13,

    "car": 14,
    "truck": 15,
    "bus": 16,
    "train": 17,

    "motorcycle": 18,
    "bicycle": 19,
}

# reverse lookup
ID_TO_CLASS = {v: k for k, v in CLASS_IDS.items()}

# ------------------------------------------------------------
# Official Cityscapes colors
# ------------------------------------------------------------

CLASS_COLORS = {
    # "unlabeled": (0, 0, 0),

    "road": (128, 64, 128),
    "sidewalk": (244, 35, 232),
    "building": (70, 70, 70),
    "wall": (102, 102, 156),
    "fence": (190, 153, 153),
    "pole": (153, 153, 153),

    "traffic light": (250, 170, 30),
    "traffic sign": (220, 220, 0),

    "vegetation": (107, 142, 35),
    "terrain": (152, 251, 152),
    "sky": (70, 130, 180),

    "person": (220, 20, 60),
    "rider": (255, 0, 0),

    "car": (0, 0, 142),
    "truck": (0, 0, 70),
    "bus": (0, 60, 100),
    "train": (0, 80, 100),

    "motorcycle": (0, 0, 230),
    "bicycle": (119, 11, 32),
}

# ------------------------------------------------------------
# Labels ignored by Cityscapes evaluation
# ------------------------------------------------------------

EXCLUDE_LABELS = {
    "ego vehicle",
    "rectification border",
    "out of roi",
    "license plate",
}

# ------------------------------------------------------------
# Palette for PIL (768 values)
# ------------------------------------------------------------

def build_palette():

    palette = []

    for idx in range(256):

        if idx in ID_TO_CLASS:

            rgb = CLASS_COLORS[ID_TO_CLASS[idx]]

        else:

            rgb = (0, 0, 0)

        palette.extend(rgb)

    return palette