"""Decoupled Attention Detection Head."""

import math
import torch.nn as nn
from .neck import DepthwiseSeparableConv, ChannelAttention, SpatialAttention


class DecoupledAttentionHead(nn.Module):
    """Decoupled detection head with separate cls/reg branches."""
    
    def __init__(self, in_channels=256, num_classes=4, num_convs=2):
        super().__init__()
        self.num_classes = num_classes
        
        # Shared stem
        self.stem = nn.Sequential(
            DepthwiseSeparableConv(in_channels, in_channels),
            ChannelAttention(in_channels)
        )
        
        # Classification branch
        cls_layers = [DepthwiseSeparableConv(in_channels, in_channels) for _ in range(num_convs)]
        self.cls_convs = nn.Sequential(*cls_layers)
        self.cls_attention = SpatialAttention()
        self.cls_pred = nn.Conv2d(in_channels, num_classes, 1)
        
        # Regression branch
        reg_layers = [DepthwiseSeparableConv(in_channels, in_channels) for _ in range(num_convs)]
        self.reg_convs = nn.Sequential(*reg_layers)
        self.reg_attention = SpatialAttention()
        self.reg_pred = nn.Conv2d(in_channels, 4, 1)
        
        # Objectness branch
        self.obj_convs = DepthwiseSeparableConv(in_channels, in_channels)
        self.obj_pred = nn.Conv2d(in_channels, 1, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_pred.bias, bias_value)
        nn.init.constant_(self.obj_pred.bias, bias_value)
    
    def forward(self, x):
        x = self.stem(x)
        
        cls_feat = self.cls_attention(self.cls_convs(x))
        cls_out = self.cls_pred(cls_feat)
        
        reg_feat = self.reg_attention(self.reg_convs(x))
        reg_out = self.reg_pred(reg_feat)
        
        obj_feat = self.obj_convs(x)
        obj_out = self.obj_pred(obj_feat)
        
        return cls_out, reg_out, obj_out


class MultiScaleHead(nn.Module):
    """Applies detection head to multiple scales."""
    
    def __init__(self, in_channels=256, num_classes=4):
        super().__init__()
        self.head = DecoupledAttentionHead(in_channels, num_classes)
        self.num_classes = num_classes
    
    def forward(self, features):
        outputs = {}
        for name, feat in features.items():
            cls_out, reg_out, obj_out = self.head(feat)
            outputs[name] = {'cls': cls_out, 'reg': reg_out, 'obj': obj_out}
        return outputs
