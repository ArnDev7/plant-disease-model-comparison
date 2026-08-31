import os
import json
import yaml
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Tuple, List, Dict, Any

from src.train import build_from_config
from src.data.transforms import build_eval_transforms

def resolve_device(requested_device: str = "auto") -> torch.device:
    """Selects CUDA if requested/available, else CPU."""
    if requested_device.lower() == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested but is not available.")
        return torch.device("cuda")
    elif requested_device.lower() == "cpu":
        return torch.device("cpu")
    else:  # auto
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_inference_model(
    checkpoint_path: str = "reports/vit_b16/checkpoints/vit_b16_best.pt",
    label_map_path: str = "data/splits/label_map.json",
    config_path: str = "configs/vit_b16.yaml",
    device: torch.device = None
) -> Tuple[torch.nn.Module, Dict[int, str], Dict[str, Any]]:
    """Loads the model, label map, and config for inference."""
    if device is None:
        device = resolve_device("auto")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Please ensure the model is trained and the checkpoint is available.")
    
    if not os.path.exists(label_map_path):
        raise FileNotFoundError(f"Label map not found at {label_map_path}.")
        
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}.")
        
    with open(label_map_path, "r") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(idx_to_class)
    
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = build_from_config(cfg, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    
    return model, idx_to_class, cfg

def predict_image(
    model: torch.nn.Module,
    idx_to_class: Dict[int, str],
    image: Image.Image,
    device: torch.device
) -> Tuple[str, float, List[str], List[float]]:
    """Runs inference on a PIL RGB image and returns predictions."""
    transforms = build_eval_transforms(img_size=224, resize_shorter=256)
    input_tensor = transforms(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1).squeeze(0)
        
    top_probs, top_indices = torch.topk(probs, 3)
    top_probs = top_probs.cpu().numpy().tolist()
    top_indices = top_indices.cpu().numpy().tolist()
    
    top_classes = [idx_to_class[idx].replace("_", " ") for idx in top_indices]
    
    predicted_class = top_classes[0]
    confidence_score = top_probs[0]
    
    return predicted_class, confidence_score, top_classes, top_probs
