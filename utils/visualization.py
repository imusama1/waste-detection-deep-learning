"""Visualization utilities."""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import sys
sys.path.append('..')
from config import CLASS_NAMES, COLORS


def visualize_prediction(model, image_path, device, conf_thresh=0.25, save_path=None):
    """Visualize model predictions on an image."""
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    
    img_resized = img.resize((640, 640))
    img_tensor = torch.from_numpy(np.array(img_resized)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = ((img_tensor - mean) / std).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        results = model.decode_predictions(outputs, conf_thresh=conf_thresh)[0]
    
    boxes = results['boxes'].cpu().numpy()
    scores = results['scores'].cpu().numpy()
    labels = results['labels'].cpu().numpy()
    
    boxes[:, [0, 2]] *= orig_w / 640
    boxes[:, [1, 3]] *= orig_h / 640
    
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)
    
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=3,
                                  edgecolor=COLORS[int(label)], facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1-10, f"{CLASS_NAMES[int(label)]}: {score:.2f}",
                color='white', fontsize=12,
                bbox=dict(boxstyle="round", facecolor=COLORS[int(label)], alpha=0.8))
    
    ax.axis('off')
    ax.set_title(f"Detections: {len(boxes)}")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
    return results


def plot_training_history(history, save_path=None):
    """Plot training history."""
    fig, ax = plt.subplots(1, figsize=(10, 5))
    ax.plot(history['train_loss'], label='Train Loss')
    ax.plot(history['val_loss'], label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.set_title('Training History')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
