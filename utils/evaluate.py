"""Evaluation script."""

import os
import argparse
import shutil
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import *
from models import CustomDetector
from utils import WasteDataset, collate_fn


def compute_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / (area1 + area2 - inter + 1e-6)


def detect_backbone_size(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'neck.lateral_p3.0.weight' in checkpoint['model_state_dict']:
        ch = checkpoint['model_state_dict']['neck.lateral_p3.0.weight'].shape[1]
        return {64: 'n', 128: 's', 192: 'm', 256: 'l'}.get(ch, 's')
    return checkpoint.get('backbone_size', 's')


def evaluate(model, dataloader):
    model.eval()
    stats = {i: {'TP': 0, 'FP': 0, 'FN': 0} for i in range(NUM_CLASSES)}
    conf_matrix = np.zeros((NUM_CLASSES + 1, NUM_CLASSES + 1), dtype=int)
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = images.to(DEVICE)
            outputs = model(images)
            results = model.decode_predictions(outputs, conf_thresh=CONF_THRESHOLD)
            
            for i, result in enumerate(results):
                pred_boxes = result['boxes'].cpu().numpy()
                pred_labels = result['labels'].cpu().numpy()
                gt_boxes = targets[i]['boxes'].numpy()
                gt_labels = targets[i]['labels'].numpy()
                
                matched = set()
                for p_idx, p_box in enumerate(pred_boxes):
                    p_label = int(pred_labels[p_idx])
                    best_iou, best_idx = 0, -1
                    
                    for g_idx, g_box in enumerate(gt_boxes):
                        iou = compute_iou(p_box, g_box)
                        if iou > best_iou:
                            best_iou, best_idx = iou, g_idx
                    
                    if best_iou >= IOU_THRESHOLD and best_idx not in matched:
                        gt_label = int(gt_labels[best_idx])
                        if p_label == gt_label:
                            stats[p_label]['TP'] += 1
                        else:
                            stats[p_label]['FP'] += 1
                        conf_matrix[gt_label][p_label] += 1
                        matched.add(best_idx)
                    else:
                        stats[p_label]['FP'] += 1
                        conf_matrix[NUM_CLASSES][p_label] += 1
                
                for g_idx, g_label in enumerate(gt_labels):
                    if g_idx not in matched:
                        stats[int(g_label)]['FN'] += 1
                        conf_matrix[int(g_label)][NUM_CLASSES] += 1
    
    return stats, conf_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=MODEL_SAVE_PATH)
    args = parser.parse_args()
    
    # Prepare data
    temp_dir = 'temp_eval'
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    all_files = sorted([f for f in os.listdir(DATA_DIR) if os.path.splitext(f)[1] in extensions])
    val_files = all_files[int(len(all_files) * 0.8):]
    
    for f in val_files:
        shutil.copy(os.path.join(DATA_DIR, f), temp_dir)
        txt = os.path.splitext(f)[0] + ".txt"
        if os.path.exists(os.path.join(DATA_DIR, txt)):
            shutil.copy(os.path.join(DATA_DIR, txt), temp_dir)
    
    # Load model
    backbone_size = detect_backbone_size(args.model)
    print(f"📦 Loading model (backbone: YOLOv8{backbone_size})")
    model = CustomDetector(NUM_CLASSES, backbone_size)
    checkpoint = torch.load(args.model, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    
    dataset = WasteDataset(temp_dir, IMG_SIZE, train=False)
    loader = DataLoader(dataset, BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    stats, conf_matrix = evaluate(model, loader)
    
    # Report
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-"*40)
    
    f1_scores = []
    for i, name in enumerate(CLASS_NAMES):
        tp, fp, fn = stats[i]['TP'], stats[i]['FP'], stats[i]['FN']
        p = tp / (tp + fp + 1e-6)
        r = tp / (tp + fn + 1e-6)
        f1 = 2 * p * r / (p + r + 1e-6)
        f1_scores.append(f1)
        print(f"{name:<10} {p:.4f}     {r:.4f}     {f1:.4f}")
    
    print("-"*40)
    print(f"Mean F1: {np.mean(f1_scores):.4f}")
    
    # Plot confusion matrix
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.figure(figsize=(10, 8))
    labels = CLASS_NAMES + ['Background']
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    cm_norm = np.divide(conf_matrix.astype(float), row_sums, where=row_sums!=0)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    plt.close()
    
    shutil.rmtree(temp_dir)
    print(f"\n✅ Results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
