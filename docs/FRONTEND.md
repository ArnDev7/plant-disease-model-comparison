# Streamlit Inference Frontend

## Purpose
This document provides details on the optional Streamlit inference frontend included in this repository. It serves as an interactive portfolio demonstration to run predictions using the academically trained model checkpoints on local leaf images.

## Requirements
- `streamlit`
- `Pillow`
- Existing deep learning environment (`torch`, `torchvision`)

## Checkpoint Placement
The application defaults to loading the best ViT-B/16 checkpoint. Ensure the checkpoint file exists exactly at:
`reports/vit_b16/checkpoints/vit_b16_best.pt`

*Note: The GitHub source repository may not include trained weights due to file size limits. You must train the model or download the checkpoint locally.*

## Input Format
- Supported file types: PNG, JPG, JPEG
- The image will be converted to RGB format.
- Preprocessing applies the exact academic evaluation transforms: resized (shorter edge 256), center-cropped to 224x224, and ImageNet normalized.

## Prediction Flow
1. Load the official configuration, label map, and best checkpoint.
2. Accept a user-uploaded image via the frontend.
3. Preprocess the image to exactly match the evaluation protocol.
4. Run model inference in `torch.no_grad()` evaluation mode.
5. Apply softmax to extract confidence scores.
6. Display the predicted class and the top-3 confidence scores.

## Device Selection
The frontend will automatically attempt to select `cuda` if available. If a compatible GPU is not found, it gracefully falls back to `cpu`. You do not need a GPU to run the inference frontend, though it may be slightly slower.

## Run Command
Start the frontend from the project root using your virtual environment:

```bash
.\.venv\Scripts\python.exe -m streamlit run app.py
```
Or, generically:
```bash
streamlit run app.py
```

## Limitations
- **Academic Demonstration Only:** This is not a professional agricultural diagnostic tool.
- **Out-of-Distribution Data:** The model is trained on PlantVillage data, which features highly controlled laboratory conditions with uniform backgrounds. Predictions on images taken "in the wild" under field conditions may not be accurate.
- **Confidence Calibration:** The softmax confidence score represents the model's output distribution, not necessarily calibrated probabilistic accuracy.

## Troubleshooting
- **Missing Checkpoint:** Ensure the `.pt` file is located exactly at the expected relative path.
- **Dependency Issues:** Ensure you have activated the correct virtual environment before installing `streamlit` or running `app.py`. Do not globally install dependencies if it overrides your CUDA PyTorch installation.
- **Invalid Images:** Corrupted or unsupported image files will be caught by the frontend and present a clear error message instead of crashing the server.
