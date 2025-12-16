"""Training script."""

import os
import json
import random
import shutil
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import *
from models import CustomDetector
from dataset import WasteDataset, collate_fn
from loss import DetectionLoss
from utils import plot_training_history


class Trainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = DetectionLoss(NUM_CLASSES, CLS_WEIGHT, REG_WEIGHT, OBJ_WEIGHT)
        
        params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        self.scaler = torch.amp.GradScaler('cuda')
        self.best_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'lr': []}
    
    def _create_scheduler(self, epochs, current_epoch=0):
        return torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=LEARNING_RATE if current_epoch < UNFREEZE_EPOCH else 5e-4,
            epochs=epochs - current_epoch,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.1
        )
    
    def train_epoch(self, scheduler):
        self.model.train()
        total_loss = 0
        
        for images, targets in tqdm(self.train_loader, desc="Training"):
            images = images.to(DEVICE)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            
            with torch.amp.autocast('cuda'):
                outputs = self.model(images)
                loss_dict = self.criterion(outputs, targets, IMG_SIZE)
                loss = loss_dict['total']
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            scheduler.step()
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self):
        self.model.train()
        total_loss = 0
        
        for images, targets in tqdm(self.val_loader, desc="Validation"):
            images = images.to(DEVICE)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            
            with torch.amp.autocast('cuda'):
                outputs = self.model(images)
                loss_dict = self.criterion(outputs, targets, IMG_SIZE)
            total_loss += loss_dict['total'].item()
        
        return total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0
    
    def train(self, epochs):
        print(f"\nTraining for {epochs} epochs...")
        scheduler = self._create_scheduler(epochs)
        
        for epoch in range(epochs):
            print(f"\n{'='*50}\nEpoch {epoch+1}/{epochs}\n{'='*50}")
            
            if epoch == UNFREEZE_EPOCH:
                print("\nUnfreezing backbone...")
                self.model.unfreeze_backbone('last')
                params = [p for p in self.model.parameters() if p.requires_grad]
                self.optimizer = torch.optim.AdamW(params, lr=5e-4, weight_decay=WEIGHT_DECAY)
                scheduler = self._create_scheduler(epochs, epoch)
            
            train_loss = self.train_epoch(scheduler)
            val_loss = self.validate()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'val_loss': val_loss,
                    'history': self.history,
                    'backbone_size': BACKBONE_SIZE
                }, MODEL_SAVE_PATH)
                print(f"Saved best model")
        
        return self.history


def main():
    print(f"Device: {DEVICE}")
    
    # Prepare data
    extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    all_files = os.listdir(DATA_DIR)
    pairs = [(f, os.path.splitext(f)[0] + ".txt") for f in all_files 
             if os.path.splitext(f)[1] in extensions and os.path.splitext(f)[0] + ".txt" in all_files]
    
    random.shuffle(pairs)
    split = int(len(pairs) * 0.8)
    train_pairs, val_pairs = pairs[:split], pairs[split:]
    
    # Create temp directories
    for d in ['temp_train', 'temp_val']:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)
    
    for img, lbl in train_pairs:
        shutil.copy(os.path.join(DATA_DIR, img), 'temp_train')
        shutil.copy(os.path.join(DATA_DIR, lbl), 'temp_train')
    for img, lbl in val_pairs:
        shutil.copy(os.path.join(DATA_DIR, img), 'temp_val')
        shutil.copy(os.path.join(DATA_DIR, lbl), 'temp_val')
    
    train_dataset = WasteDataset('temp_train', IMG_SIZE, train=True)
    val_dataset = WasteDataset('temp_val', IMG_SIZE, train=False)
    
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    # Train
    model = CustomDetector(NUM_CLASSES, BACKBONE_SIZE, FREEZE_BACKBONE)
    trainer = Trainer(model, train_loader, val_loader)
    history = trainer.train(EPOCHS)
    
    # Save and plot
    with open('training_history.json', 'w') as f:
        json.dump(history, f)
    plot_training_history(history, 'training_history.png')
    
    # Cleanup
    shutil.rmtree('temp_train')
    shutil.rmtree('temp_val')
    print("\nDone!")


if __name__ == "__main__":
    main()
