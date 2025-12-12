"""Dataset and DataLoader utilities."""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance


class WasteDataset(Dataset):
    """Dataset for waste detection with augmentation."""
    
    def __init__(self, data_dir, img_size=640, train=True):
        self.data_dir = data_dir
        self.img_size = img_size
        self.train = train
        self.images = []
        self.labels = []
        
        extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        for f in os.listdir(data_dir):
            if os.path.splitext(f)[1] in extensions:
                label_path = os.path.join(data_dir, os.path.splitext(f)[0] + ".txt")
                if os.path.exists(label_path):
                    self.images.append(os.path.join(data_dir, f))
                    self.labels.append(label_path)
        
        print(f"📂 Found {len(self.images)} samples in {data_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def augment(self, img, boxes):
        """Apply augmentation."""
        aug_boxes = boxes.copy() if boxes else []
        
        # Horizontal flip
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if aug_boxes:
                aug_boxes = [[self.img_size - b[2], b[1], self.img_size - b[0], b[3]] for b in aug_boxes]
        
        # Color jitter
        if random.random() > 0.3:
            for enhancer_cls in [ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color]:
                img = enhancer_cls(img).enhance(random.uniform(0.7, 1.3))
        
        # Rotation
        if random.random() > 0.7:
            img = img.rotate(random.uniform(-15, 15), fillcolor=(128, 128, 128))
        
        return img, aug_boxes
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        img = img.resize((self.img_size, self.img_size))
        
        boxes, labels = [], []
        with open(self.labels[idx], 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    xc, yc, w, h = [float(x) * self.img_size for x in parts[1:]]
                    boxes.append([xc - w/2, yc - h/2, xc + w/2, yc + h/2])
                    labels.append(cls_id)
        
        if self.train:
            img, boxes = self.augment(img, boxes)
        
        # To tensor and normalize
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            'labels': torch.tensor(labels, dtype=torch.long) if labels else torch.zeros(0, dtype=torch.long)
        }
        return img_tensor, target


def collate_fn(batch):
    """Custom collate function for detection."""
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets
