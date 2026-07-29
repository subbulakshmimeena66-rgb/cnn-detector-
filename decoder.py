
import torch
from src.config import IMAGE_WIDTH, IMAGE_HEIGHT


def decode_predictions(predictions, threshold=0.5):
    """
    predictions: raw model output, shape [B, 5] (no sigmoid applied yet)
                 -> [objectness, xmin_norm, ymin_norm, xmax_norm, ymax_norm]
    Returns: list of dicts, one per image in the batch:
             {"has_object", "confidence", "xmin", "ymin", "xmax", "ymax"}
    """
    predictions = torch.sigmoid(predictions)  # squash all values to [0, 1]

    boxes = []
    for i in range(predictions.shape[0]):
        obj_score = predictions[i, 0].item()

        xmin = predictions[i, 1].item() * IMAGE_WIDTH
        ymin = predictions[i, 2].item() * IMAGE_HEIGHT
        xmax = predictions[i, 3].item() * IMAGE_WIDTH
        ymax = predictions[i, 4].item() * IMAGE_HEIGHT

        boxes.append({
            "has_object": obj_score > threshold,
            "confidence": obj_score,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax
        })

    return boxes