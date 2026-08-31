import streamlit as st
import os
import sys
from pathlib import Path
from PIL import Image

# Add project root to sys.path to allow src imports
sys.path.append(str(Path(__file__).parent))

from src.inference import load_inference_model, predict_image, resolve_device

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
def get_model_and_labels():
    device = resolve_device("auto")
    model, idx_to_class, cfg = load_inference_model(
        CHECKPOINT_PATH, LABEL_MAP_PATH, CONFIG_PATH, device
    )
    return model, idx_to_class, cfg, device

# Setup UI
st.title("🌿 Plant Disease Classifier")
st.write("Predictions are based on 15 PlantVillage crop-disease classes.")
st.caption("Disclaimer: This is an academic image-classification demonstration, not agricultural or professional diagnostic advice. Images from PlantVillage may not represent field conditions.")

try:
    model, idx_to_class, cfg, device = get_model_and_labels()
except Exception as e:
    st.error(str(e))
    st.stop()

# Optional Automated Sample Test Section
summary_path = Path("reports/frontend_test/summary.json")
if summary_path.exists():
    with st.expander("Automated Sample Test", expanded=False):
        import json
        with open(summary_path, "r") as f:
            summary = json.load(f)
        st.write("### Automated frontend sample-test results")
        st.write("*These values are not the official full test-set metrics.*")
        st.write(f"- **Tested samples:** {summary['total_successfully_processed']}")
        st.write(f"- **Sample Top-1 accuracy:** {summary['sample_top1_accuracy'] * 100:.2f}%")
        st.write(f"- **Sample Top-3 accuracy:** {summary['sample_top3_accuracy'] * 100:.2f}%")
        st.write(f"- **Mean inference time:** {summary['mean_inference_time_ms']:.2f} ms")
        st.write(f"- **Device used:** {summary['device']}")

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
        
        predicted_class, confidence, top_classes, top_probs = predict_image(
            model, idx_to_class, image, device
        )
        
        st.write(f"**Predicted Class:** {predicted_class}")
        st.write(f"**Model Confidence Score:** {confidence * 100:.2f}%")
        
        st.write("### Top-3 Predictions")
        for i in range(3):
            st.write(f"{i+1}. {top_classes[i]}: {top_probs[i] * 100:.2f}%")
            st.progress(float(top_probs[i]))

