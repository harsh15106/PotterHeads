import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION (THE CONTROL PANEL)
# ==========================================
# FIXED: Points exactly to the folder in your screenshot
DATA_DIR = r"F:\Startathon\Offroad_Segmentation_Training_Dataset" 

CLASSES = 10        
BATCH_SIZE = 8      
NUM_WORKERS = 4     
EPOCHS = 40         
LR = 0.0001         
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"--- STARTING TRAINING SESSION ---")
print(f"Device: {DEVICE} | Batch Size: {BATCH_SIZE} | Epochs: {EPOCHS}")

# ==========================================
# 2. AUGMENTATION PIPELINES
# ==========================================
def get_training_augmentation():
    return A.Compose([
        A.Resize(512, 512),                 
        A.HorizontalFlip(p=0.5),            
        A.VerticalFlip(p=0.5),              
        A.RandomRotate90(p=0.5),            
        A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=15, shift_limit=0.1, p=0.5),
        A.RandomBrightnessContrast(p=0.2),  
        A.GaussNoise(p=0.1),                
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_validation_augmentation():
    return A.Compose([
        A.Resize(512, 512),                 
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

# ==========================================
# 3. DATASET CLASS (THE LOADER)
# ==========================================
class DesertDataset(Dataset):
    def __init__(self, root_dir, split="train", augmentation=None):
        self.root_dir = root_dir
        self.split = split
        self.augmentation = augmentation
        
        # FIXED: Using "Color_Images" and "Segmentation" based on your screenshots
        self.images_dir = os.path.join(root_dir, split, "Color_Images")
        self.masks_dir = os.path.join(root_dir, split, "Segmentation")
        
        # Debug check
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"CRITICAL ERROR: Could not find {self.images_dir}. Check folder names!")
            
        self.images = os.listdir(self.images_dir)
        
        # Mapping 100->1, 200->2...
        self.mapping = {
            0: 0, 100: 1, 200: 2, 300: 3, 500: 4, 
            550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
        }

    def __len__(self):
        return len(self.images)

    def mask_to_class_id(self, mask):
        mask_out = np.zeros_like(mask)
        for k, v in self.mapping.items():
            mask_out[mask == k] = v
        return mask_out

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)

        # Load Image and Mask
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, -1)
        
        # Map Class IDs
        mask = self.mask_to_class_id(mask)

        # Apply Augmentation
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask.long()

# ==========================================
# 4. TRAINING LOOP
# ==========================================
def main():
    # --- A. SETUP DATA ---
    train_dataset = DesertDataset(DATA_DIR, split="train", augmentation=get_training_augmentation())
    valid_dataset = DesertDataset(DATA_DIR, split="val", augmentation=get_validation_augmentation())
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0) # Only True if workers > 0
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0)
    )

    print(f"Loaded {len(train_dataset)} Training Images and {len(valid_dataset)} Validation Images.")

    # --- B. SETUP MODEL ---
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",        
        encoder_weights="imagenet",     
        in_channels=3, 
        classes=CLASSES, 
    ).to(DEVICE)

    # --- C. OPTIMIZER & LOSS ---
    # FIXED: Combine Dice + Focal Loss to force the model to learn small objects
    dice_loss = smp.losses.DiceLoss(mode='multiclass')
    focal_loss = smp.losses.FocalLoss(mode='multiclass')
    loss_fn = lambda x, y: dice_loss(x, y) + focal_loss(x, y)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = GradScaler() 
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_iou = 0.0

    # --- D. THE LOOP ---
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        # Training
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for images, masks in loop:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()
            
            with autocast():
                logits = model(images)
                loss = loss_fn(logits, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        # --- E. VALIDATION (THE FIXED PART) ---
        model.eval()
        val_loss = 0
        
        # We will store the IoU for every batch here
        iou_scores = []
        
        with torch.no_grad():
            for images, masks in valid_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                
                with autocast():
                    logits = model(images)
                    loss = loss_fn(logits, masks)
                    val_loss += loss.item()
                
                # Convert raw outputs to class predictions (0-9)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                # Calculate True Positives, False Positives, etc.
                tp, fp, fn, tn = smp.metrics.get_stats(
                    preds.long(), 
                    masks.long(), 
                    mode='multiclass', 
                    num_classes=CLASSES
                )
                
                # Calculate IoU for this batch (micro-average is standard for hackathons)
                batch_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
                iou_scores.append(batch_iou.item())
        
        # --- F. METRICS CALCULATION ---
        scheduler.step()
        
        # 1. Average the scores (This replaces the broken "intersection / union" line)
        final_val_iou = np.mean(iou_scores)
        avg_val_loss = val_loss / len(valid_loader)

        print(f"\nResults Epoch {epoch+1}:")
        print(f"  > Val Loss: {avg_val_loss:.4f}")
        print(f"  > Val IoU:  {final_val_iou:.4f}")  # This will be a REAL number now (e.g. 0.35)

        # Save Best Model
        if final_val_iou > best_iou:
            best_iou = final_val_iou
            torch.save(model.state_dict(), "best_model_final.pth")
            print(f"  >>> NEW HIGH SCORE! Model Saved.")
        
        print("-" * 40)

if __name__ == "__main__":
    main()