import torch
from src.config import IMAGE_WIDTH, IMAGE_HEIGHT


def create_detection_target(boxes):
    """
    Returns a single target vector of shape [5]:
    [objectness, xmin_norm, ymin_norm, xmax_norm, ymax_norm]
    Assumes one box per image (DFUC-style). If multiple boxes exist,
    the first one is used.
    """
    target = torch.zeros(5, dtype=torch.float32)

    if len(boxes) == 0:
        return target  # no ulcer -> all zeros, objectness = 0

    box = boxes.iloc[0]  # take first box (single-ulcer assumption)

    xmin = float(box["xmin"]) / IMAGE_WIDTH
    ymin = float(box["ymin"]) / IMAGE_HEIGHT
    xmax = float(box["xmax"]) / IMAGE_WIDTH
    ymax = float(box["ymax"]) / IMAGE_HEIGHT

    target[0] = 1.0
    target[1] = xmin
    target[2] = ymin
    target[3] = xmax
    target[4] = ymax

    return target
