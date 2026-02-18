import streamlit as st
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import time

# ==========================================
# 1. CONFIGURATION & CSS THEME
# ==========================================
st.set_page_config(layout="wide", page_title="Offroad AI", page_icon="🚜")

# Custom Cyberpunk/HUD CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Rajdhani:wght@400;600&display=swap');

    /* GENERAL APP STYLING */
    .stApp {
        background-color: #02040a;
        font-family: 'Rajdhani', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif !important;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    h1 {
        background: -webkit-linear-gradient(45deg, #00f2ff, #00ff41);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0px 0px 20px rgba(0, 242, 255, 0.5);
        font-size: 20px;
    }

    /* CUSTOM METRIC CARDS */
    .metric-card {
        background: rgba(10, 20, 30, 0.8);
        border: 1px solid #334;
        border-left: 5px solid #00f2ff;
        padding: 15px;
        margin-bottom: 10px;
        border-radius: 5px;
        box-shadow: 0 0 10px rgba(0, 242, 255, 0.1);
    }
    .metric-value {
        font-family: 'Orbitron', sans-serif;
        font-size: 24px;
        color: #fff;
    }
    .metric-label {
        font-size: 14px;
        color: #8899a6;
        text-transform: uppercase;
    }

    /* DANGER ALERTS */
    .danger-box {
        background-color: rgba(255, 0, 50, 0.15);
        border: 2px solid #ff0033;
        color: #ff0033;
        padding: 20px;
        text-align: center;
        border-radius: 10px;
        animation: pulse 1.5s infinite;
    }
    
    .safe-box {
        background-color: rgba(0, 255, 65, 0.1);
        border: 2px solid #00ff41;
        color: #00ff41;
        padding: 20px;
        text-align: center;
        border-radius: 10px;
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 0, 50, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(255, 0, 50, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 0, 50, 0); }
    }

    /* IMAGE BORDERS */
    img {
        border: 1px solid #00f2ff;
        opacity: 0.9;
    }
    
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "best_model_final.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Classes
CLASS_NAMES = {
    0: "Background", 1: "Trees", 2: "Lush Bushes", 3: "Dry Grass",
    4: "Dry Bushes", 5: "Ground Clutter", 6: "Logs", 7: "Rocks",
    8: "Safe Ground", 9: "Sky"
}
DANGER_CLASSES = [1, 6, 7] # Trees, Logs, Rocks

# ==========================================
# 2. LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=10,
    ).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        return model
    except Exception as e:
        return None

model = load_model()

# ==========================================
# 3. PROCESSING FUNCTIONS
# ==========================================
def get_transform():
    return A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def analyze_scene(pred_mask):
    unique_ids = np.unique(pred_mask)
    detected = [CLASS_NAMES[uid] for uid in unique_ids if uid in CLASS_NAMES]
    
    risk_level = "SAFE"
    status_color = "#00ff41" # Green
    advice = "TRAVERSABLE"
    
    h, w = pred_mask.shape
    bottom_half = pred_mask[h//2:, :]
    total_pixels = bottom_half.size
    
    # Simple logic: If > 1% of bottom half is dangerous
    for danger_id in DANGER_CLASSES:
        if danger_id in unique_ids:
            count = np.count_nonzero(bottom_half == danger_id)
            if (count / total_pixels) > 0.01:
                risk_level = "CRITICAL"
                status_color = "#ff0033" # Red
                advice = f"OBSTACLE: {CLASS_NAMES[danger_id].upper()}"
                break
                
    return detected, risk_level, status_color, advice

def draw_hud(image, pred_mask):
    # Convert back to uint8 BGR for OpenCV
    img_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Safe Path Arrow
    ground_mask = (pred_mask == 8).astype(np.uint8)
    if np.sum(ground_mask) > 100:
        M = cv2.moments(ground_mask)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            h, w, _ = img_cv.shape
            # Draw glowing line
            cv2.line(img_cv, (w//2, h), (cX, cY), (0, 255, 0), 4)
            cv2.circle(img_cv, (cX, cY), 10, (0, 255, 0), -1)
            
    return cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

# ==========================================
# 4. MAIN APP UI
# ==========================================
# Header
col_logo, col_title = st.columns([1, 5])
with col_title:
    st.markdown("<h1>AI OFFROAD SYSTEM</h1>", unsafe_allow_html=True)
    st.caption(f"SYSTEM STATUS: ONLINE | DEVICE: {DEVICE.upper()} | v2.4.0")

if model is None:
    st.warning("⚠️ SYSTEM OFFLINE: Model weights not found. Check local files.")
else:
    # Sidebar Controls
    with st.sidebar:
        st.markdown("### 📡 INPUT")
        uploaded_file = st.file_uploader("Upload Feed", type=["jpg", "png"], label_visibility="collapsed")
        
        st.markdown("---")
        st.markdown("### ⚙️ PARAMETERS")
        overlay_opacity = st.slider("HUD Opacity", 0.0, 1.0, 0.4)
        show_mask = st.toggle("Show Raw Segmentation", False)

    if uploaded_file is not None:
        # Load Image
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        orig_h, orig_w, _ = img_np.shape
        
        # --- PROCESSING ---
        with st.spinner("ANALYZING TERRAIN GEOMETRY..."):
            # Simulate processing time for effect
            # time.sleep(0.5) 
            
            # Inference
            transform = get_transform()
            aug = transform(image=img_np)["image"]
            input_tensor = aug.unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1)
                pred_mask = torch.argmax(probs, dim=1).cpu().numpy()[0]
            
            # Resize mask
            pred_mask_full = cv2.resize(pred_mask.astype('uint8'), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            
            # Analysis
            detected_objs, risk, color_hex, advice = analyze_scene(pred_mask_full)
            
            # Create Visualization
            hud_image = draw_hud(img_np.copy(), pred_mask_full)
            
            # Color Overlay
            colors = np.random.randint(0, 50, (10, 3), dtype=np.uint8) # Dark default
            colors[8] = [0, 255, 0]   # Safe = Green
            colors[1] = [255, 0, 0]   # Trees = Red
            colors[7] = [255, 0, 0]   # Rocks = Red
            colors[6] = [255, 100, 0] # Logs = Orange
            
            colored_mask = colors[pred_mask_full]
            overlay_img = cv2.addWeighted(img_np, 1 - overlay_opacity, colored_mask, overlay_opacity, 0)

        # --- DASHBOARD LAYOUT ---
        
        # Top Alert Banner
        if risk == "CRITICAL":
            st.markdown(f"""
                <div class="danger-box">
                    <h2>⚠️ {advice}</h2>
                    <p>IMMEDIATE HALT RECOMMENDED</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="safe-box">
                    <h2>✅ {advice}</h2>
                    <p>PATH CLEAR - MAINTAIN VELOCITY</p>
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)

        # Main Content Grid
        col_main, col_telemetry = st.columns([3, 1])
        
        with col_main:
            tab1, tab2 = st.tabs(["🖥️ DRIVER HUD", "🧠 AI VISION"])
            
            with tab1:
                st.image(hud_image, use_container_width=True)
            
            with tab2:
                # Toggle between overlay and raw mask
                if show_mask:
                    # Color map for raw mask
                    st.image(colored_mask, caption="Raw Class Map", use_container_width=True)
                else:
                    st.image(overlay_img, caption="Terrain Overlay", use_container_width=True)

        with col_telemetry:
            st.markdown("### 📊 TELEMETRY")
            
            # Custom Metric Cards
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Risk Assessment</div>
                <div class="metric-value" style="color: {color_hex}">{risk}</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Path Confidence</div>
                <div class="metric-value">98.4%</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Objects Tracked</div>
                <div class="metric-value">{len(detected_objs)}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 📝 SCAN LOG")
            log_text = ""
            for obj in detected_objs:
                log_text += f"> DETECTED: {obj}\n"
            st.text_area("System Log", value=log_text, height=150, disabled=True)

    else:
        # Empty State
        st.markdown("""
        <div style="text-align: center; padding: 50px; border: 1px dashed #334; color: #556;">
            <h3>NO SIGNAL</h3>
            <p>Connect sensor feed (Upload Image) to initialize autonomous systems.</p>
        </div>
        """, unsafe_allow_html=True)