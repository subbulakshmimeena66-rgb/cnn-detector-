import torch
import torch.nn as nn


class DetectionLoss(nn.Module):
    def __init__(self, box_loss_weight=5.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.box_loss_weight = box_loss_weight

    def forward(self, predictions, targets):
        """
        predictions: [B, 5] raw (no sigmoid applied yet)
        targets:     [B, 5] -> [objectness, xmin, ymin, xmax, ymax]
        """
        pred_obj = predictions[:, 0]
        pred_box = predictions[:, 1:]

        target_obj = targets[:, 0]
        target_box = targets[:, 1:]

        object_loss = self.bce(pred_obj, target_obj)

        # only penalize box coords where an object actually exists
        mask = target_obj.bool()
        if mask.sum() > 0:
            pred_box_sig = torch.sigmoid(pred_box[mask])
            box_loss = nn.functional.l1_loss(pred_box_sig, target_box[mask])
        else:
            box_loss = torch.tensor(0.0, device=predictions.device)

        total_loss = object_loss + self.box_loss_weight * box_loss

        return total_loss, object_loss, box_loss