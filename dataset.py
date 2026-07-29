import os
import cv2
import pandas as pd

from src.config import IMAGE_WIDTH, IMAGE_HEIGHT

from src.preprocessing import (
    resize_image,
    normalize_image,
    apply_clahe,
    apply_bilateral_filter,
    resize_bounding_boxes
)

def load_annotations(annotation_path):
    return pd.read_csv(annotation_path)

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def get_boxes(annotations, image_name):
    image_id = int(image_name.replace(".jpg", ""))

    boxes = annotations[
        annotations["filename"] == image_id
    ].copy()

    return boxes


def prepare_training_sample(image_name,
                            image_folder,
                            annotations):

    image_path = os.path.join(image_folder, image_name)

    image = load_image(image_path)

    original_height, original_width = image.shape[:2]

    image = resize_image(image)

    image = normalize_image(image)

    image = (image * 255).astype("uint8")

    image = apply_clahe(image)

    image = apply_bilateral_filter(image)

    x_scale = IMAGE_WIDTH / original_width
    y_scale = IMAGE_HEIGHT / original_height
    
    
    boxes = get_boxes(
        annotations,
        image_name
    )

    boxes = resize_bounding_boxes(
        boxes,
        x_scale,
        y_scale
    )
    return image, boxes

import torch
from torch.utils.data import Dataset

from src.targets import create_detection_target


class DFUCDataset(Dataset):

    def __init__(self, image_path, annotations):
        self.image_path = image_path
        self.annotations = annotations
        self.filenames = annotations["filename"].astype(str).unique()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index] + ".jpg"

        image, boxes = prepare_training_sample(
            filename,
            self.image_path,
            self.annotations
        )

        target = create_detection_target(boxes)

        image = torch.from_numpy(image).permute(2, 0, 1).float() /255.0

        return image, target
    

