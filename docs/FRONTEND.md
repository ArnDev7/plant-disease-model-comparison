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

## Automated Inference Testing

### Purpose
The automated inference test validates the frontend prediction pipeline in batch mode. It simulates processing a small number of deterministic samples to ensure that the identical inference code path processes them correctly, catching pipeline errors and visual misclassifications without needing manual UI interaction.

### Default Sample Strategy
- **Split**: Uses the `test` split to avoid overlapping with training data.
- **Quantity**: Runs 3 samples per class, capping at 45 images.
- **Determinism**: A fixed seed (`--seed 42`) guarantees the exact same subset of images is chosen across runs.

### Commands
Run the default batch test:
```bash
.\.venv\Scripts\python.exe scripts\test_inference.py --samples-per-class 3 --split test --seed 42 --device auto --output-dir reports/frontend_test
```

Run specific images manually:
```bash
.\.venv\Scripts\python.exe scripts\test_inference.py --specific-images "data/raw/PlantVillage/Tomato_healthy/image1.JPG"
```

### Outputs
Outputs are saved in `reports/frontend_test/` (excluded from Git tracking):
- `summary.json`: High-level test summary and timing.
- `predictions.csv`: Image-by-image tabular prediction log.
- `class_summary.csv`: Accuracy broken down by class.
- `prediction_gallery*.png`: Visual layout of the selected images overlaid with their predicted class and ground truth.
- `confidence_summary.png`: A histogram plotting the model's confidence distribution for correct vs. incorrect classifications.

### Disclaimer
**Important**: The sample-test results from this script test only the functional operation of the Streamlit application pipeline on ~45 images. They **do not** replace the official full test-set metrics (~3,000 images) located in the main `reports/` directory.

### Behavior Notes
- **CUDA Behavior**: Setting `--device cuda` explicitly will fail if CUDA is not available. The default `--device auto` seamlessly falls back to CPU if necessary.
- **Misclassification**: A misclassification by the model does not fail the script. Misclassification is a valid functional outcome of the test.
