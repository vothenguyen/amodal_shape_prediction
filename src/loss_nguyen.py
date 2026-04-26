import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskAmodalLoss(nn.Module):
    def __init__(self, weight_v=1.0, weight_b=1.5, weight_p=0.5, weight_a=2.0):
        super().__init__()
        self.wv = weight_v
        self.wb = weight_b
        self.wp = weight_p
        self.wa = weight_a
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def dice_loss(self, pred, target, smooth=1.0):
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum(dim=(2, 3))
        union = pred_sigmoid.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        loss = 1 - (2. * intersection + smooth) / (union + smooth)
        return loss.mean()
        
    def forward(self, preds, targets):
        """
        preds: tuple (Ma, Mv, Mb, Mp)
        targets: tuple (GT_v, GT_b, GT_a)
        """
        Ma, Mv, Mb, Mp = preds
        GT_v, GT_b, GT_a = targets
        
        # Resize targets to match predictions if necessary
        # Assume spatial sizes match
        
        # 1. Supervise Visible Mask
        loss_v = self.bce_loss(Mv, GT_v) + self.dice_loss(Mv, GT_v)
        
        # 2. Supervise Boundary Mask
        loss_b = self.bce_loss(Mb, GT_b) + self.dice_loss(Mb, GT_b)
        
        # 3. Supervise Shape Prior
        # The prior tries to emulate GT_a from Codebook
        loss_p = self.bce_loss(Mp, GT_a)
        
        # 4. Supervise final Amodal Mask
        loss_a = self.bce_loss(Ma, GT_a) + self.dice_loss(Ma, GT_a)
        
        total_loss = self.wv * loss_v + self.wb * loss_b + self.wp * loss_p + self.wa * loss_a
        
        return {
            'total_loss': total_loss,
            'loss_v': loss_v,
            'loss_b': loss_b,
            'loss_p': loss_p,
            'loss_a': loss_a
        }
