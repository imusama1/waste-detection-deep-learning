"""Configuration settings for the waste detection model."""

import torch

# Paths
DATA_DIR = "Dataset"
OUTPUT_DIR = "outputs"
MODEL_SAVE_PATH = "best_custom_model.pth"

# Model settings
NUM_CLASSES = 4
CLASS_NAMES = ['AluCan', 'Glass', 'HDPEM', 'PET']
BACKBONE_SIZE = 'm'  # Options: 'n', 's', 'm', 'l', 'x'
FREEZE_BACKBONE = True

# Training settings
BATCH_SIZE = 8
EPOCHS = 30
IMG_SIZE = 640
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 0.01
UNFREEZE_EPOCH = 15

# Loss weights
CLS_WEIGHT = 1.5
REG_WEIGHT = 5.0
OBJ_WEIGHT = 1.0

# Inference settings
CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45
IOU_THRESHOLD = 0.5

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Colors for visualization
COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
