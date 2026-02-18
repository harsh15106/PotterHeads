import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Path to your TEST images (Update if needed based on your screenshots)
TEST_IMAGES_DIR = r"F:\Startathon\Offroad_Segmentation_testImages\Color_Images"

# Where to save the results
OUTPUT_DIR = r"F:\Startathon\Test_Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = "best_model_final.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Class Definitions (10 Classes)
CLASS_NAMES = {
    0: "Background",
    1: "Trees",
    2: "Lush Bushes",
    3: "Dry Grass",
    4: "Dry Bushes",
    5: "Ground Clutter",
    6: "Logs",
    7: "Rocks",           
    8: "Safe Ground",     
    9: "Sky"
}

# Dangerous Objects (Stop/Avoid these)
DANGER_CLASSES = [1, 6, 7] # Trees, Logs, Rocks

print(f"--- STARTING CO-PILOT INFERENCE ---")
print(f"Reading from: {TEST_IMAGES_DIR}")
print(f"Saving to:    {OUTPUT_DIR}")
print(f"Device:       {DEVICE}")

# ==========================================
# 2. SETUP MODEL & TRANSFORM
# ==========================================
def get_test_transform():
    return A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

# Load the trained model
model = smp.UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=10,
).to(DEVICE)

try:
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    print("✔ Model loaded successfully!")
except FileNotFoundError:
    print(f"❌ Error: Could not find '{MODEL_PATH}'. Did you finish training?")
    exit()

# ==========================================
# 3. ANALYSIS FUNCTIONS (The Brains)
# ==========================================
def analyze_scene(pred_mask):
    """
    Scans the prediction to find objects and calculate risk.
    """
    unique_ids = np.unique(pred_mask)
    detected_objects = []
    risk_level = "SAFE"
    recommendation = "Proceed at Speed"
    color = (0, 255, 0) # Green

    # 1. Detect Objects
    for uid in unique_ids:
        if uid in CLASS_NAMES:
            detected_objects.append(CLASS_NAMES[uid])

    # 2. Safety Check
    # We look for danger classes in the bottom half of the image (the path)
    height, width = pred_mask.shape
    bottom_half = pred_mask[height//2:, :]
    
    total_pixels = bottom_half.size
    
    for danger_id in DANGER_CLASSES:
        if danger_id in unique_ids:
            # Count how many danger pixels are in the bottom half
            count = np.count_nonzero(bottom_half == danger_id)
            percentage = count / total_pixels
            
            # If obstacle covers > 1% of the road view, trigger warning
            if percentage > 0.01:
                risk_level = "OBSTACLE DETECTED"
                obj_name = CLASS_NAMES[danger_id].upper()
                recommendation = f"AVOID {obj_name}!"
                color = (0, 0, 255) # Red
                break

    return detected_objects, risk_level, recommendation, color

def draw_hud(image, pred_mask):
    """
    Draws the 'Co-Pilot' overlay: Path Arrow + Text Info.
    """
    # 1. Analyze
    objects, risk, advice, status_color = analyze_scene(pred_mask)
    
    # 2. Draw Safe Path (Green Arrow)
    # Assuming Class 8 is 'Safe Ground' (Update ID if your mapping is different)
    ground_mask = (pred_mask == 8).astype(np.uint8) 
    
    if np.sum(ground_mask) > 100: # If we see enough ground
        M = cv2.moments(ground_mask)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            h, w, _ = image.shape
            car_center = (w // 2, h)
            
            # Draw Arrow
            cv2.arrowedLine(image, car_center, (cX, cY), (0, 255, 0), 8, tipLength=0.2)
    
    # 3. Draw Dashboard Box
    # Semi-transparent black box
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (450, 160), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
    # 4. Add Text
    cv2.putText(image, f"STATUS: {risk}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.putText(image, f"ACTION: {advice}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # List objects
    obj_str = ", ".join(objects[:3]) # First 3 objects
    if len(objects) > 3: obj_str += "..."
    cv2.putText(image, f"Vis: {obj_str}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    return image

# ==========================================
# 4. MAIN INFERENCE LOOP
# ==========================================
def main():
    if not os.path.exists(TEST_IMAGES_DIR):
        print(f"❌ Error: Test folder not found: {TEST_IMAGES_DIR}")
        return

    images_list = os.listdir(TEST_IMAGES_DIR)
    print(f"Found {len(images_list)} images. Processing...")

    transform = get_test_transform()

    for img_name in tqdm(images_list):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(TEST_IMAGES_DIR, img_name)
        
        # 1. Load Image
        image = cv2.imread(img_path)
        if image is None: continue
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w, _ = original_image.shape

        # 2. Preprocess
        augmented = transform(image=original_image)["image"]
        input_tensor = augmented.unsqueeze(0).to(DEVICE)

        # 3. Predict
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_mask = torch.argmax(probs, dim=1).cpu().numpy()[0]

        # 4. Resize mask back to original size
        pred_mask_resized = cv2.resize(pred_mask.astype('uint8'), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # 5. Draw Co-Pilot HUD
        result_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR) # Back to BGR for OpenCV
        final_output = draw_hud(result_image, pred_mask_resized)

        # 6. Save
        save_path = os.path.join(OUTPUT_DIR, f"analyzed_{img_name}")
        cv2.imwrite(save_path, final_output)

    print(f"\n✅ Processing Complete!")
    print(f"Go check your results here: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()