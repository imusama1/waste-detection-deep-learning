"""Inference script."""

import os
import argparse
import torch
import cv2
import numpy as np
from PIL import Image

from config import *
from models import CustomDetector


def detect_backbone_size(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'neck.lateral_p3.0.weight' in checkpoint['model_state_dict']:
        ch = checkpoint['model_state_dict']['neck.lateral_p3.0.weight'].shape[1]
        return {64: 'n', 128: 's', 192: 'm', 256: 'l'}.get(ch, 's')
    return checkpoint.get('backbone_size', 's')


def load_model(model_path):
    backbone_size = detect_backbone_size(model_path)
    print(f"📦 Loading model (backbone: YOLOv8{backbone_size})")
    model = CustomDetector(NUM_CLASSES, backbone_size)
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()
    return model


def preprocess(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = image.resize((IMG_SIZE, IMG_SIZE))
    tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return ((tensor - mean) / std).unsqueeze(0).to(DEVICE)


def draw_boxes(image, boxes, labels, scores):
    colors = [(255, 107, 107), (78, 205, 196), (69, 183, 209), (150, 206, 180)]
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = map(int, box)
        color = colors[int(label)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        text = f"{CLASS_NAMES[int(label)]}: {score:.2f}"
        cv2.putText(image, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image


def predict_image(model, image_path, save_path=None):
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    
    tensor = preprocess(img)
    with torch.no_grad():
        outputs = model(tensor)
        results = model.decode_predictions(outputs, CONF_THRESHOLD, NMS_THRESHOLD)[0]
    
    boxes = results['boxes'].cpu().numpy()
    boxes[:, [0, 2]] *= orig_w / IMG_SIZE
    boxes[:, [1, 3]] *= orig_h / IMG_SIZE
    
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img_cv = draw_boxes(img_cv, boxes, results['labels'].cpu().numpy(), results['scores'].cpu().numpy())
    
    if save_path:
        cv2.imwrite(save_path, img_cv)
        print(f"✅ Saved to {save_path}")
    else:
        cv2.imshow("Prediction", img_cv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return results


def predict_folder(model, folder_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    images = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1] in extensions]
    
    print(f"📂 Processing {len(images)} images...")
    for img_file in images:
        img_path = os.path.join(folder_path, img_file)
        save_path = os.path.join(output_folder, f"pred_{img_file}")
        predict_image(model, img_path, save_path)
    print(f"✅ Results saved to {output_folder}/")


def predict_webcam(model):
    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        orig_h, orig_w = frame.shape[:2]
        tensor = preprocess(frame)
        
        with torch.no_grad():
            outputs = model(tensor)
            results = model.decode_predictions(outputs, CONF_THRESHOLD, NMS_THRESHOLD)[0]
        
        boxes = results['boxes'].cpu().numpy()
        if len(boxes) > 0:
            boxes[:, [0, 2]] *= orig_w / IMG_SIZE
            boxes[:, [1, 3]] *= orig_h / IMG_SIZE
            frame = draw_boxes(frame, boxes, results['labels'].cpu().numpy(), results['scores'].cpu().numpy())
        
        cv2.imshow("Waste Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Waste Detection Inference")
    parser.add_argument('--model', default=MODEL_SAVE_PATH, help='Model path')
    parser.add_argument('--image', type=str, help='Single image path')
    parser.add_argument('--folder', type=str, help='Folder of images')
    parser.add_argument('--output', type=str, default='predictions', help='Output folder')
    parser.add_argument('--webcam', action='store_true', help='Use webcam')
    parser.add_argument('--conf', type=float, default=CONF_THRESHOLD, help='Confidence threshold')
    args = parser.parse_args()
    
    global CONF_THRESHOLD
    CONF_THRESHOLD = args.conf
    
    model = load_model(args.model)
    
    if args.webcam:
        predict_webcam(model)
    elif args.image:
        predict_image(model, args.image)
    elif args.folder:
        predict_folder(model, args.folder, args.output)
    else:
        print("Please specify --image, --folder, or --webcam")


if __name__ == "__main__":
    main()
