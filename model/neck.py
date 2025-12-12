"""Custom ABFP (Adaptive Bidirectional Feature Pyramid) Neck."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention."""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial attention module."""
    
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(concat))


class DepthwiseSeparableConv(nn.Module):
    """Efficient depthwise separable convolution."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                   stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class AdaptiveFusionBlock(nn.Module):
    """Adaptive feature fusion with learnable weights."""
    
    def __init__(self, channels):
        super().__init__()
        self.weight1 = nn.Parameter(torch.ones(1))
        self.weight2 = nn.Parameter(torch.ones(1))
        self.conv = DepthwiseSeparableConv(channels, channels)
        self.attention = ChannelAttention(channels)
    
    def forward(self, x1, x2):
        w1 = torch.exp(self.weight1) / (torch.exp(self.weight1) + torch.exp(self.weight2))
        w2 = torch.exp(self.weight2) / (torch.exp(self.weight1) + torch.exp(self.weight2))
        fused = w1 * x1 + w2 * x2
        fused = self.conv(fused)
        fused = self.attention(fused)
        return fused


class CrossScaleConnection(nn.Module):
    """Cross-scale feature connection with residual path."""
    
    def __init__(self, in_channels, out_channels, scale_factor=2, mode='up'):
        super().__init__()
        self.mode = mode
        self.scale_factor = scale_factor
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dwconv = DepthwiseSeparableConv(out_channels, out_channels)
        self.spatial_att = SpatialAttention()
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        if self.mode == 'up':
            x = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        else:
            x = F.max_pool2d(x, kernel_size=self.scale_factor, stride=self.scale_factor)
        residual = x
        x = self.dwconv(x)
        x = self.spatial_att(x)
        x = x + residual
        return x


class CustomNeck_ABFP(nn.Module):
    """Adaptive Bidirectional Feature Pyramid Network."""
    
    def __init__(self, in_channels=[128, 256, 512], out_channels=256):
        super().__init__()
        
        # Lateral connections
        self.lateral_p5 = nn.Sequential(
            nn.Conv2d(in_channels[2], out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.SiLU(inplace=True))
        self.lateral_p4 = nn.Sequential(
            nn.Conv2d(in_channels[1], out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.SiLU(inplace=True))
        self.lateral_p3 = nn.Sequential(
            nn.Conv2d(in_channels[0], out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.SiLU(inplace=True))
        
        # Top-down path
        self.up_p5_to_p4 = CrossScaleConnection(out_channels, out_channels, 2, 'up')
        self.fusion_td_p4 = AdaptiveFusionBlock(out_channels)
        self.up_p4_to_p3 = CrossScaleConnection(out_channels, out_channels, 2, 'up')
        self.fusion_td_p3 = AdaptiveFusionBlock(out_channels)
        
        # Bottom-up path
        self.down_p3_to_p4 = CrossScaleConnection(out_channels, out_channels, 2, 'down')
        self.fusion_bu_p4 = AdaptiveFusionBlock(out_channels)
        self.down_p4_to_p5 = CrossScaleConnection(out_channels, out_channels, 2, 'down')
        self.fusion_bu_p5 = AdaptiveFusionBlock(out_channels)
        
        # Cross-scale skip connections
        self.cross_p5_to_p3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 4, 1, bias=False),
            nn.BatchNorm2d(out_channels // 4), nn.SiLU(inplace=True))
        self.cross_p3_to_p5 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 4, 1, bias=False),
            nn.BatchNorm2d(out_channels // 4), nn.SiLU(inplace=True))
        
        # Final refinement
        self.refine_p3 = nn.Sequential(
            nn.Conv2d(out_channels + out_channels // 4, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.SiLU(inplace=True))
        self.refine_p4 = DepthwiseSeparableConv(out_channels, out_channels)
        self.refine_p5 = nn.Sequential(
            nn.Conv2d(out_channels + out_channels // 4, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.SiLU(inplace=True))
        
        # Output attention
        self.out_att_p3 = ChannelAttention(out_channels)
        self.out_att_p4 = ChannelAttention(out_channels)
        self.out_att_p5 = ChannelAttention(out_channels)
    
    def forward(self, features):
        p3, p4, p5 = features['p3'], features['p4'], features['p5']
        
        # Lateral connections
        p5_lat = self.lateral_p5(p5)
        p4_lat = self.lateral_p4(p4)
        p3_lat = self.lateral_p3(p3)
        
        # Top-down
        p5_up = self.up_p5_to_p4(p5_lat)
        p4_td = self.fusion_td_p4(p4_lat, p5_up)
        p4_up = self.up_p4_to_p3(p4_td)
        p3_td = self.fusion_td_p3(p3_lat, p4_up)
        
        # Bottom-up
        p3_down = self.down_p3_to_p4(p3_td)
        p4_bu = self.fusion_bu_p4(p4_td, p3_down)
        p4_down = self.down_p4_to_p5(p4_bu)
        p5_bu = self.fusion_bu_p5(p5_lat, p4_down)
        
        # Cross-scale
        p5_to_p3 = F.interpolate(self.cross_p5_to_p3(p5_bu), size=p3_td.shape[2:], mode='nearest')
        p3_to_p5 = F.adaptive_max_pool2d(self.cross_p3_to_p5(p3_td), output_size=p5_bu.shape[2:])
        
        # Refinement
        p3_out = self.refine_p3(torch.cat([p3_td, p5_to_p3], dim=1))
        p4_out = self.refine_p4(p4_bu)
        p5_out = self.refine_p5(torch.cat([p5_bu, p3_to_p5], dim=1))
        
        # Output attention
        p3_out = self.out_att_p3(p3_out)
        p4_out = self.out_att_p4(p4_out)
        p5_out = self.out_att_p5(p5_out)
        
        return {'p3': p3_out, 'p4': p4_out, 'p5': p5_out}
