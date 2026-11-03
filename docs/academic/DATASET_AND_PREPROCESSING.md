# Dataset and Preprocessing

- **Dataset Name**: PlantVillage
- **Raw dataset path expected by loader**: `data/raw/PlantVillage/`
- **Total sample count**: 20,638
- **Number of target classes**: 15
- **Training sample count**: 14,446
- **Validation sample count**: 3,096
- **Test sample count**: 3,096
- **Split percentages**: 70% Training / 15% Validation / 15% Test
- **Split-loading mechanism**: Precomputed mapping via `data/splits/split.csv`
- **Stratification behavior**: Stratified by disease class during split generation
- **Random seed**: 42
- **Class mapping mechanism**: Deterministic mapping via `data/splits/label_map.json`
- **Input image dimensions**: 224x224
- **Training transforms**: RandomResizedCrop (scale 0.8-1.0), RandomHorizontalFlip (p=0.5), RandomRotation (±15 degrees), ColorJitter (brightness, contrast, saturation=0.2, hue=0.05), ToTensor, Normalize
- **Validation/Test transforms**: Resize shorter side to 256, CenterCrop to 224, ToTensor, Normalize
- **ImageNet normalization values**: Mean = [0.485, 0.456, 0.406], Std = [0.229, 0.224, 0.225]
- **DataLoader behavior**: PyTorch `DataLoader` with parallel workers configured via YAML.
- **Training drop_last behavior**: `drop_last=True` for training loader.
- **Validation/Test sample preservation**: Iterates all samples sequentially without dropping any batch (`drop_last=False`).
- **Data integrity checks completed**: Split counts verified against total, consistent classes across all loaders.

## Preprocessing Pipeline Summary
1. **Load**: Images are loaded from disk.
2. **Augment (Train only)**: Images are randomly cropped, resized, horizontally flipped, rotated, and color-jittered to improve generalization.
3. **Resize/Crop (Val/Test only)**: Images are resized and center-cropped to 224x224 for consistent evaluation without randomness.
4. **Tensor Conversion**: Images are converted to PyTorch tensors in the [0, 1] range.
5. **Normalize**: Channel-wise normalization is applied using standard ImageNet mean and standard deviation.
