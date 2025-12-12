"""YOLOv8 Backbone Extractor."""

import torch.nn as nn


class YOLOv8Backbone(nn.Module):
    """Extracts multi-scale features from pretrained YOLOv8."""
    
    # Output channels for each YOLOv8 variant
    SIZE_CHANNELS = {
        'n': [64, 128, 256],
        's': [128, 256, 512],
        'm': [192, 384, 576],
        'l': [256, 512, 512],
        'x': [320, 640, 640]
    }
    
    def __init__(self, model_size='s', freeze=True):
        super().__init__()
        
        from ultralytics import YOLO
        yolo = YOLO(f'yolov8{model_size}.pt')
        backbone = yolo.model.model
        
        # Extract backbone stages
        self.stage1 = nn.Sequential(*[backbone[i] for i in range(5)])
        self.stage2 = nn.Sequential(*[backbone[i] for i in range(5, 7)])
        self.stage3 = nn.Sequential(*[backbone[i] for i in range(7, 10)])
        
        self.out_channels = self.SIZE_CHANNELS.get(model_size, [128, 256, 512])
        self.model_size = model_size
        
        if freeze:
            self.freeze()
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        print("   🔒 Backbone frozen")
    
    def unfreeze(self, stages='all'):
        if stages == 'all':
            for param in self.parameters():
                param.requires_grad = True
            print("   🔓 Backbone fully unfrozen")
        elif stages == 'last':
            for param in self.stage3.parameters():
                param.requires_grad = True
            print("   🔓 Backbone stage3 unfrozen")
    
    def forward(self, x):
        p3 = self.stage1(x)
        p4 = self.stage2(p3)
        p5 = self.stage3(p4)
        return {'p3': p3, 'p4': p4, 'p5': p5}
