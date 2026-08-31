import streamlit as st
import torch
import json
from PIL import Image
import torch.nn.functional as F
import yaml
import os
import sys
from pathlib import Path

# Add project root to sys.path to allow src imports
sys.path.append(str(Path(__file__).parent))

from src.train import build_from_config
from src.data.transforms import build_eval_transforms

# Configure Streamlit page
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="wide"
)

# Constants
CHECKPOINT_PATH = "reports/vit_b16/checkpoints/vit_b16_best.pt"
LABEL_MAP_PATH = "data/splits/label_map.json"
CONFIG_PATH = "configs/vit_b16.yaml"

@st.cache_resource
def load_model_and_labels():
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}. Please ensure the model is trained and the checkpoint is available.")
    
    if not os.path.exists(LABEL_MAP_PATH):
        raise FileNotFoundError(f"Label map not found at {LABEL_MAP_PATH}.")
        
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config not found at {CONFIG_PATH}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(LABEL_MAP_PATH, "r") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(idx_to_class)
    
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    
    model = build_from_config(cfg, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    
    return model, idx_to_class, device, cfg

# Setup UI
st.title("🌿 Plant Disease Classifier")
st.write("Predictions are based on 15 PlantVillage crop-disease classes.")
st.caption("Disclaimer: This is an academic image-classification demonstration, not agricultural or professional diagnostic advice. Images from PlantVillage may not represent field conditions.")

try:
    model, idx_to_class, device, cfg = load_model_and_labels()
except Exception as e:
    st.error(str(e))
    st.stop()

st.sidebar.title("Model Information")
st.sidebar.write(f"**Model Name:** ViT-B/16")
st.sidebar.write(f"**Architecture Family:** Vision Transformer / Self-Attention")
st.sidebar.write(f"**Input Size:** 224x224")
st.sidebar.write(f"**Number of Classes:** {len(idx_to_class)}")
st.sidebar.write(f"**Execution Device:** {str(device).upper()}")

uploaded_file = st.file_uploader("Upload a plant leaf image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Invalid or corrupted image: {e}")
        st.stop()
        
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
    with col2:
        st.subheader("Prediction Result")
        
        transforms = build_eval_transforms(img_size=224, resize_shorter=256)
        input_tensor = transforms(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1).squeeze(0)
            
        top_probs, top_indices = torch.topk(probs, 3)
        top_probs = top_probs.cpu().numpy()
        top_indices = top_indices.cpu().numpy()
        
        predicted_class = idx_to_class[top_indices[0]].replace("_", " ")
        confidence = top_probs[0] * 100
        
        st.write(f"**Predicted Class:** {predicted_class}")
        st.write(f"**Model Confidence Score:** {confidence:.2f}%")
        
        st.write("### Top-3 Predictions")
        for i in range(3):
            cls_name = idx_to_class[top_indices[i]].replace("_", " ")
            score = top_probs[i] * 100
            st.write(f"{i+1}. {cls_name}: {score:.2f}%")
            st.progress(float(top_probs[i]))
