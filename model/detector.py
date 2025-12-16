import torch
import torch.nn as nn
from .backbone import YOLOv8Backbone
from .neck import CustomNeck_ABFP
from .head import MultiScaleHead


class CustomDetector(nn.Module):
    """Complete object detection model."""
    
    def __init__(self, num_classes=4, backbone_size='m', freeze_backbone=True):
        super().__init__()
        self.num_classes = num_classes
        
        self.backbone = YOLOv8Backbone(backbone_size, freeze=freeze_backbone)
        self.neck = CustomNeck_ABFP(in_channels=self.backbone.out_channels, out_channels=256)
        self.head = MultiScaleHead(in_channels=256, num_classes=num_classes)
        self.strides = {'p3': 8, 'p4': 16, 'p5': 32}
        
        self._print_info(backbone_size, freeze_backbone)
    
    def _print_info(self, backbone_size, freeze_backbone):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"   CustomDetector initialized:")
        print(f"   Backbone: YOLOv8{backbone_size} ({'frozen' if freeze_backbone else 'trainable'})")
        print(f"   Neck: ABFP | Head: Decoupled Attention")
        print(f"   Params: {total:,} total, {trainable:,} trainable")
    
    def unfreeze_backbone(self, stages='all'):
        self.backbone.unfreeze(stages)
    
    def forward(self, x):
        features = self.backbone(x)
        features = self.neck(features)
        outputs = self.head(features)
        return outputs
    
    def decode_predictions(self, outputs, conf_thresh=0.5, nms_thresh=0.45):
        """Decode model outputs to bounding boxes."""
        from torchvision.ops import nms
        
        all_boxes, all_scores, all_labels = [], [], []
        
        for scale_name, scale_output in outputs.items():
            cls_pred = scale_output['cls'].sigmoid()
            reg_pred = scale_output['reg']
            obj_pred = scale_output['obj'].sigmoid()
            
            B, _, H, W = cls_pred.shape
            stride = self.strides[scale_name]
            
            yv, xv = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
            grid = torch.stack([xv, yv], dim=-1).to(cls_pred.device).float()
            
            scores = obj_pred * cls_pred
            max_scores, labels = scores.max(dim=1)
            
            reg_pred = reg_pred.permute(0, 2, 3, 1)
            xy = (grid + reg_pred[..., :2].sigmoid()) * stride
            wh = reg_pred[..., 2:4].exp() * stride
            boxes = torch.cat([xy - wh / 2, xy + wh / 2], dim=-1)
            
            all_boxes.append(boxes.reshape(B, -1, 4))
            all_scores.append(max_scores.reshape(B, -1))
            all_labels.append(labels.reshape(B, -1))
        
        all_boxes = torch.cat(all_boxes, dim=1)
        all_scores = torch.cat(all_scores, dim=1)
        all_labels = torch.cat(all_labels, dim=1)
        
        batch_results = []
        for i in range(all_boxes.shape[0]):
            boxes, scores, labels = all_boxes[i], all_scores[i], all_labels[i]
            
            mask = scores > conf_thresh
            boxes, scores, labels = boxes[mask], scores[mask], labels[mask]
            
            if len(boxes) > 0:
                keep_boxes, keep_scores, keep_labels = [], [], []
                for cls_id in range(self.num_classes):
                    cls_mask = labels == cls_id
                    if cls_mask.sum() == 0:
                        continue
                    cls_boxes, cls_scores = boxes[cls_mask], scores[cls_mask]
                    keep_idx = nms(cls_boxes, cls_scores, nms_thresh)
                    keep_boxes.append(cls_boxes[keep_idx])
                    keep_scores.append(cls_scores[keep_idx])
                    keep_labels.append(torch.full((len(keep_idx),), cls_id, device=boxes.device))
                
                if keep_boxes:
                    boxes = torch.cat(keep_boxes)
                    scores = torch.cat(keep_scores)
                    labels = torch.cat(keep_labels)
                else:
                    boxes = torch.zeros((0, 4), device=all_boxes.device)
                    scores = torch.zeros(0, device=all_boxes.device)
                    labels = torch.zeros(0, dtype=torch.long, device=all_boxes.device)
            else:
                boxes = torch.zeros((0, 4), device=all_boxes.device)
                scores = torch.zeros(0, device=all_boxes.device)
                labels = torch.zeros(0, dtype=torch.long, device=all_boxes.device)
            
            batch_results.append({'boxes': boxes, 'scores': scores, 'labels': labels})
        
        return batch_results
