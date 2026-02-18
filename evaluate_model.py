import torch
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import cv2
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = r"F:\Startathon\Offroad_Segmentation_Training_Dataset"
MODEL_PATH = "best_model_final.pth"
DEVICE = "cuda"
BATCH_SIZE = 8

# Mapping from your dataset
CLASS_NAMES = ["Background", "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes", 
               "Ground Clutter", "Logs", "Rocks", "Safe Ground", "Sky"]
CLASSES = 10

# ==========================================
# SETUP (Reuse logic from train script)
# ==========================================
def get_validation_augmentation():
    return A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

class DesertDataset(Dataset):
    def __init__(self, root_dir, split="val", augmentation=None):
        self.root_dir = root_dir
        self.split = split
        self.augmentation = augmentation
        self.images_dir = os.path.join(root_dir, split, "Color_Images")
        self.masks_dir = os.path.join(root_dir, split, "Segmentation")
        self.images = os.listdir(self.images_dir)
        self.mapping = {0: 0, 100: 1, 200: 2, 300: 3, 500: 4, 550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9}

    def __len__(self): return len(self.images)
    def mask_to_class_id(self, mask):
        mask_out = np.zeros_like(mask)
        for k, v in self.mapping.items(): mask_out[mask == k] = v
        return mask_out
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.images[idx])
        image = cv2.imread(img_path); image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, -1); mask = self.mask_to_class_id(mask)
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        return image, mask.long()

# ==========================================
# EVALUATION LOGIC
# ==========================================
def main():
    print("--- STARTING DETAILED EVALUATION ---")
    
    # Load Data
    valid_dataset = DesertDataset(DATA_DIR, split="val", augmentation=get_validation_augmentation())
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Load Model
    model = smp.UnetPlusPlus(encoder_name="resnet34", in_channels=3, classes=CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    # Metrics
    tp_total = [0] * CLASSES
    fp_total = [0] * CLASSES
    fn_total = [0] * CLASSES
    
    with torch.no_grad():
        for images, masks in tqdm(valid_loader):
            images = images.to(DEVICE); masks = masks.to(DEVICE)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            # Calculate stats per class
            tp, fp, fn, tn = smp.metrics.get_stats(preds.long(), masks.long(), mode='multiclass', num_classes=CLASSES)
            
            for i in range(CLASSES):
                tp_total[i] += tp[:, i].sum().item()
                fp_total[i] += fp[:, i].sum().item()
                fn_total[i] += fn[:, i].sum().item()

    # Calculate IoU per class
    ious = []
    print("\n=== PER CLASS PERFORMANCE ===")
    print(f"{'CLASS NAME':<20} | {'IoU SCORE':<10}")
    print("-" * 35)
    
    for i in range(CLASSES):
        iou = (tp_total[i] + 1e-6) / (tp_total[i] + fp_total[i] + fn_total[i] + 1e-6)
        ious.append(iou)
        print(f"{CLASS_NAMES[i]:<20} | {iou:.4f}")
        
    print("-" * 35)
    print(f"MEAN IoU: {np.mean(ious):.4f}")

    # Plot Bar Chart
    plt.figure(figsize=(10, 6))
    plt.barh(CLASS_NAMES, ious, color='skyblue')
    plt.xlabel('IoU Score')
    plt.title('Model Accuracy by Object Type')
    plt.xlim(0, 1)
    plt.grid(axis='x')
    plt.savefig('class_performance.png')
    print("✅ Saved detailed chart to 'class_performance.png'")

if __name__ == "__main__":
    main()