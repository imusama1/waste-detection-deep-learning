"""Loss functions for object detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pred_sigmoid = torch.sigmoid(pred)
        p_t = pred_sigmoid * target + (1 - pred_sigmoid) * (1 - target)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_weight = self.alpha * target + (1 - self.alpha) * (1 - target)
        return (alpha_weight * focal_weight * bce).mean()


class DetectionLoss(nn.Module):
    """Combined detection loss."""
    
    # UPDATED: Changed cls_weight to 2.5 and obj_weight to 0.5
    def __init__(self, num_classes=4, cls_weight=2.5, reg_weight=5.0, obj_weight=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.focal_loss = FocalLoss()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.obj_weight = obj_weight
    
    def forward(self, outputs, targets, img_size):
        device = next(iter(outputs.values()))['cls'].device
        total_cls = torch.tensor(0.0, device=device)
        total_reg = torch.tensor(0.0, device=device)
        total_obj = torch.tensor(0.0, device=device)
        
        strides = {'p3': 8, 'p4': 16, 'p5': 32}
        
        for scale_name, out in outputs.items():
            cls_pred, reg_pred, obj_pred = out['cls'], out['reg'], out['obj']
            B, _, H, W = cls_pred.shape
            stride = strides[scale_name]
            
            cls_target = torch.zeros_like(cls_pred)
            obj_target = torch.zeros_like(obj_pred)
            reg_target = torch.zeros_like(reg_pred)
            reg_mask = torch.zeros((B, 1, H, W), device=device)
            
            for b in range(min(B, len(targets))):
                boxes, labels = targets[b]['boxes'], targets[b]['labels']
                if len(boxes) == 0:
                    continue
                
                cx = (boxes[:, 0] + boxes[:, 2]) / 2 / stride
                cy = (boxes[:, 1] + boxes[:, 3]) / 2 / stride
                gx, gy = cx.long().clamp(0, W-1), cy.long().clamp(0, H-1)
                
                for i, (x, y, label) in enumerate(zip(gx, gy, labels)):
                    cls_target[b, label, y, x] = 1.0
                    obj_target[b, 0, y, x] = 1.0
                    w = (boxes[i, 2] - boxes[i, 0]).clamp(min=1.0)
                    h = (boxes[i, 3] - boxes[i, 1]).clamp(min=1.0)
                    reg_target[b, 0, y, x] = cx[i] - x.float()
                    reg_target[b, 1, y, x] = cy[i] - y.float()
                    reg_target[b, 2, y, x] = torch.log(w / stride)
                    reg_target[b, 3, y, x] = torch.log(h / stride)
                    reg_mask[b, 0, y, x] = 1.0
            
            total_cls += self.focal_loss(cls_pred, cls_target)
            
            # UPDATED: Reduced pos_weight from 2.0 to 1.5
            bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5], device=device))
            total_obj += bce(obj_pred, obj_target)
            
            if reg_mask.sum() > 0:
                pos_mask = reg_mask.expand_as(reg_pred).bool()
                total_reg += F.smooth_l1_loss(reg_pred[pos_mask], reg_target[pos_mask])
        
        total = self.cls_weight * total_cls + self.reg_weight * total_reg + self.obj_weight * total_obj
        return {'total': total, 'cls': total_cls, 'reg': total_reg, 'obj': total_obj}
