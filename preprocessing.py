from src.config import IMAGE_SIZE

import cv2
import numpy as np


def resize_image(image, size=IMAGE_SIZE):
    return cv2.resize(image, size)


def normalize_image(image):
    return image.astype(np.float32) / 255.0


def apply_clahe(image, clip_limit=2.0, grid_size=(8, 8)):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=grid_size
    )

    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))

    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def apply_bilateral_filter(image):
    return cv2.bilateralFilter(
        image,
        d=9,
        sigmaColor=75,
        sigmaSpace=75
    )


def resize_bounding_boxes(boxes, x_scale, y_scale):
    boxes = boxes.copy()

    boxes["xmin"] = (boxes["xmin"] * x_scale).astype(int)
    boxes["ymin"] = (boxes["ymin"] * y_scale).astype(int)
    boxes["xmax"] = (boxes["xmax"] * x_scale).astype(int)
    boxes["ymax"] = (boxes["ymax"] * y_scale).astype(int)

    return boxes


def draw_boxes(image, boxes, color=(255, 0, 0), thickness=2):
    output = image.copy()

    for _, row in boxes.iterrows():
        cv2.rectangle(
            output,
            (int(row["xmin"]), int(row["ymin"])),
            (int(row["xmax"]), int(row["ymax"])),
            color,
            thickness
        )

    return output