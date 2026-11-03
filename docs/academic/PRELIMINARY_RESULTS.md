# Preliminary Results

## Official Four-Model Comparison Table
| Model | Test Accuracy | Macro Precision | Macro Recall | Macro-F1 | Top-3 Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MLP** | 77.42% | 77.36% | 71.27% | 72.07% | 94.54% |
| **Custom CNN** | 76.94% | 78.84% | 71.76% | 70.81% | 91.63% |
| **ResNet-18** | 88.53% | 88.64% | 86.97% | 87.06% | 98.29% |
| **ViT-B/16** | 99.77% | 99.79% | 99.80% | 99.79% | 100.00% |

## Additional Reference Benchmark
| Model | Test Accuracy | Macro Precision | Macro Recall | Macro-F1 | Top-3 Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **EfficientNet-B0** | 82.69% | 81.13% | 81.10% | 79.89% | 95.51% |

## Metric Rankings
- **Test accuracy**: 1. ViT-B/16 (99.77%), 2. ResNet-18 (88.53%), 3. MLP (77.42%), 4. Custom CNN (76.94%)
- **Macro-F1**: 1. ViT-B/16 (99.79%), 2. ResNet-18 (87.06%), 3. MLP (72.07%), 4. Custom CNN (70.81%)
- **Top-3 accuracy**: 1. ViT-B/16 (100.00%), 2. ResNet-18 (98.29%), 3. MLP (94.54%), 4. Custom CNN (91.63%)

## Observations
- **Percentage-point differences**: ViT-B/16 outperformed ResNet-18 by a substantial +11.24 percentage points in accuracy and +12.73 percentage points in Macro-F1.
- **Accuracy versus macro-F1 gaps**: MLP and Custom CNN show slightly larger gaps between Accuracy (~77%) and Macro-F1 (~71-72%), suggesting some class imbalances in their predictive capabilities. ViT-B/16 maintained extreme consistency (Accuracy 99.77% vs Macro-F1 99.79%).
- **Factual Description**: 
  - The parameter-heavy MLP and lightweight Custom CNN performed similarly as from-scratch baselines. 
  - ResNet-18 provided strong improvements, representing a robust convolutional transfer learning baseline. 
  - ViT-B/16 exhibited exceptional predictive capability with near-perfect classification on this specific dataset partition.
- **References**: Visualizations corresponding to these results are available under `docs/academic/assets/` (e.g., `accuracy_f1_comparison.png`, `all_metrics_comparison.png`).
