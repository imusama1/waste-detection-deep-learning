# Waste Object Detection Using Deep Learning  
### Academic Project Report Summary

This project focuses on detecting waste objects in images using deep learning.
The goal is to identify four recyclable material categories to support
automated waste sorting:

- **AluCan** (aluminium cans)  
- **Glass**  
- **HDPEM** (HDPE plastic bottles)  
- **PET** (PET plastic bottles)

We evaluated three different models:

1. **Custom ABFP Model** (main model)  
2. **ResNet50 Detector**  
3. **Pretrained YOLOv8** (reference baseline)

This README summarizes the dataset, models, evaluation metrics, and qualitative
results from the academic project.

---

## Dataset Overview

- Total images: **4,811**  
- Number of classes: **4**  
- Train/Val split: **80% / 20%**  
- Annotation format: **YOLO bounding boxes**  
- Images include variations in lighting, backgrounds, and object scales.

---

## Models Evaluated

### 1. Custom ABFP Model (Main Model)

- Pretrained YOLOv8s backbone (**frozen**)  
- Adaptive Bidirectional Feature Pyramid (**ABFP**) neck  
- Decoupled attention-based detection heads  
- Focal Loss, Smooth L1 Loss, and BCEWithLogitsLoss  
- Mixed-precision FP16 training  
- Trained for **50 epochs**

This model serves as the main experimental architecture developed for the project.

---

### 2. ResNet50 Detector

- End-to-end training  
- High precision and recall  
- Strong and stable performance across all classes  

---

### 3. Pretrained YOLOv8 (Reference Model)

- Used only as a **non-finetuned baseline**  
- Trained originally on COCO  
- Achieves excellent detection due to large-scale pretrained features  

---

## Evaluation Metrics (Updated)

| Model | mAP@0.5 | Precision | Recall |
|-------|---------|-----------|--------|
| **Custom ABFP Model** | **0.6340** | **0.5768** | **0.7269** |
| **ResNet50 Detector** | 0.90+ | 0.90–1.00 | 0.85–1.00 |
| **YOLOv8 (Pretrained)** | 0.994 | 0.991 | 0.992 |

### Observations

- The ABFP model achieves **high recall**, meaning it detects most objects.  
- Precision is **moderate**, indicating some false positives remain.  
- YOLOv8 and ResNet50 outperform the custom model, as expected.  
- PET remains the hardest class due to reflective and irregular shapes.  

---

## Confidence Curves (Yolov8)

### F1-Score vs Confidence  
<p align="center"><img src="figures/BoxF1_curve.png" width="50%"></p>

### Precision vs Confidence  
<p align="center"><img src="figures/BoxP_curve.png" width="50%"></p>

### Recall vs Confidence  
<p align="center"><img src="figures/BoxR_curve.png" width="50%"></p>

### Precision–Recall Curve  
<p align="center"><img src="figures/BoxPR_curve.png" width="50%"></p>

These curves show that the model maintains high recall across confidence
thresholds while precision drops earlier, indicating overprediction tendencies.

---

## Confusion Matrices

### Custom ABFP Model  
<p align="center"><img src="figures/confusion_matrix_custom_model.png" width="50%"></p>

### YOLOv8 (Reference Model)  
<p align="center"><img src="figures/confusion_matrix_yolo.png" width="50%"></p>

### ResNet50 Detector  
<p align="center"><img src="figures/confusion_matrix_resnet50.png" width="50%"></p>

Interpretation:

- ABFP struggles mainly with AluCan and PET.  
- ResNet50 provides strong separation across all classes.  
- YOLOv8 (pretrained) shows near-perfect confusion matrix behavior.  

---

## Qualitative Predictions

### Validation Batch Examples Yolov8

<p align="center"><img src="figures/val_batch0_labels.jpg" width="50%"></p>
<p align="center"><img src="figures/val_batch0_pred.jpg" width="50%"></p>

<p align="center"><img src="figures/val_batch1_labels.jpg" width="50%"></p>
<p align="center"><img src="figures/val_batch1_pred.jpg" width="50%"></p>

<p align="center"><img src="figures/val_batch2_labels.jpg" width="50%"></p>
<p align="center"><img src="figures/val_batch2_pred.jpg" width="50%"></p>

These provide a clear view of the model's behavior on unseen validation images.

---

## Individual Predictions ResNet50

**PET example:**  
<p align="center"><img src="figures/pred_val_00000.jpg" width="40%"></p>

**Glass example:**  
<p align="center"><img src="figures/pred_val_00001.jpg" width="40%"></p>

**AluCan example:**  
<p align="center"><img src="figures/pred_val_00002.jpg" width="40%"></p>

**HDPEM example:**  
<p align="center"><img src="figures/pred_val_00003.jpg" width="40%"></p>

---

## Qualitative Results: Custom ABFP Model

Below are example predictions showing performance across HDPEM, Glass, PET, and
AluCan in varied lighting and backgrounds.

### ABFP Predictions

<table align="center">
<tr>
<td align="center"><img src="figures/custom1.jpeg" width="45%"><br>HDPEM Detection</td>
<td align="center"><img src="figures/custom2.jpeg" width="45%"><br>Glass Detection</td>
</tr>
<tr>
<td align="center"><img src="figures/custom3.jpeg" width="45%"><br>AluCan Detection</td>
<td align="center"><img src="figures/custom4.jpeg" width="45%"><br>Additional Example</td>
</tr>
</table>

These visualisations illustrate how the custom model handles different objects
and backgrounds.

---

## Key Insights

- **ABFP** achieves strong recall but moderate precision → model detects most objects but occasionally overpredicts.  
- **ResNet50** achieves the best balance of accuracy and stability.  
- **YOLOv8** performs best overall, consistent with large-scale pretrained models.  
- PET remains the hardest class due to reflections and shape variability.  
- More training data, backbone fine-tuning, and ablations would improve performance.

---

## Conclusion

Deep learning models can effectively detect waste objects, but performance varies
depending on model architecture and training strategy. While the custom ABFP
model demonstrates that lightweight feature pyramids and attention mechanisms are
feasible for waste detection, it is outperformed by stronger pretrained
architectures like ResNet50 and YOLOv8.

This README provides a concise academic summary of the full project report and
highlights the most important findings and results.

---
