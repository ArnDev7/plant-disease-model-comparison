# Reproducibility Guide

This guide documents the procedures and configurations required to replicate the experimental results of the PlantVillage deep learning architecture comparison.

---

## 1. Environment Requirements

The codebase requires Python 3.9+ and PyTorch with torchvision.

### Dependency Installation

```bash
pip install -r requirements.txt
```

### Hardware & Acceleration Verification

The training framework automatically detects available acceleration devices (`cuda`, `mps`, or `cpu`). To verify GPU acceleration support:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count()); print('Device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

> **Note on PyTorch Installation**: If GPU acceleration is required, ensure that the PyTorch build matches your system's CUDA runtime (refer to [official PyTorch installation instructions](https://pytorch.org/get-started/locally/)).

---

## 2. Dataset Setup & Split Integrity

### Dataset Directory

Download the PlantVillage dataset (15 target classes) and place it under:

```text
data/raw/PlantVillage/
```

### Deterministic Split Preservation

To ensure identical evaluation across all model architectures, use the precomputed 70% / 15% / 15% train/val/test split files. **Do not regenerate these split files.**

You can verify split file integrity using SHA-256 checksums:

```bash
# split.csv (Expected: dd08ee9abf08eed5de3c383e9677e03e153d06e99c8c8f07762073126baea66b)
# label_map.json (Expected: d64c109735c9bd9977d1c43fa3ecbb37247f14f8f21e5fc0a7c3cc6fe84bc752)
python -c "import hashlib; print('split.csv:', hashlib.sha256(open('data/splits/split.csv', 'rb').read()).hexdigest()); print('label_map.json:', hashlib.sha256(open('data/splits/label_map.json', 'rb').read()).hexdigest())"
```

---

## 3. Training Commands

Execute each model experiment using its YAML configuration file:

### Official Academic Models

1. **MLP Baseline** (Trained from scratch):
   ```bash
   python -m src.train --config configs/mlp.yaml
   ```
   *Output Directory:* `reports/mlp/`

2. **Custom CNN Baseline** (Trained from scratch):
   ```bash
   python -m src.train --config configs/cnn_custom.yaml
   ```
   *Output Directory:* `reports/cnn_custom/`

3. **ResNet-18** (ImageNet Transfer Learning):
   ```bash
   python -m src.train --config configs/resnet18.yaml
   ```
   *Output Directory:* `reports/resnet18/` (or archived in `reports_archive/reports_20260126_143813_resnet18/`)

4. **ViT-B/16** (ImageNet Transfer Learning):
   ```bash
   python -m src.train --config configs/vit_b16.yaml
   ```
   *Output Directory:* `reports/vit_b16/`

### Additional Reference Benchmark

5. **EfficientNet-B0** (ImageNet Transfer Learning):
   ```bash
   python -m src.train --config configs/efficientnet_b0.yaml
   ```
   *Output Directory:* `reports/efficientnet_b0/` (or archived in `reports_archive/reports_20260126_151938_efficientnet/`)

---

## 4. Evaluation & Visualization Commands

### Model Evaluation

Evaluate the globally best checkpoint (selected by validation macro-F1) on the 3,096-sample test set:

```bash
python -m src.eval --config configs/mlp.yaml
python -m src.eval --config configs/cnn_custom.yaml
python -m src.eval --config configs/resnet18.yaml
python -m src.eval --config configs/vit_b16.yaml
python -m src.eval --config configs/efficientnet_b0.yaml
```

### Result Visualization

Generate per-model training curves, normalized confusion matrices, and misclassified galleries:

```bash
python -m src.visualize --normalize_cm --make_gallery --config configs/mlp.yaml
python -m src.visualize --normalize_cm --make_gallery --config configs/cnn_custom.yaml
python -m src.visualize --normalize_cm --make_gallery --config configs/resnet18.yaml
python -m src.visualize --normalize_cm --config configs/vit_b16.yaml
```

---

## 5. Cross-Model Comparison Generation

Run the unified comparison framework to aggregate test metrics, perform schema normalization, generate comparison tables (CSV, Markdown, LaTeX), and render multi-model comparison charts:

```bash
python -m src.compare
```

*Output Directory:* `reports/comparison/`

---

## 6. Reproducibility Limitations

While the data partitions, random seeds, and optimization hyperparameters are fixed, minor numerical variations may occur due to:

1. **Hardware & CUDA Non-determinism**: GPU atomic operations and CUDA convolution algorithms can introduce minor floating-point divergence.
2. **Pretrained Checkpoint Releases**: `torchvision` model weights (`ViT_B_16_Weights.DEFAULT`, `ResNet18_Weights.DEFAULT`) are fetched from official PyTorch mirrors.
3. **Execution Device**: Running on CPU vs. CUDA vs. Apple Silicon (MPS) uses different backend floating-point kernels.
