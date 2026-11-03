# Interim Technical Summary

## Project Title
Comparative Analysis of Deep Learning Architectures for Plant Disease Classification

## Problem Statement
Plant disease identification is a critical computer vision task. There is a need to systematically evaluate how different deep learning architectural paradigms perform on agricultural imagery under a standardized experimental protocol, minimizing confounding variables like data splitting and metric inconsistency.

## Project Motivation
Evaluating deep learning models across disparate publications often suffers from inconsistent dataset splits, varying image resolutions, and conflicting metric definitions. This project seeks to provide a rigorous, empirical benchmarking study to analyze the trade-offs between dense networks, CNNs, residual networks, and vision transformers.

## Dataset Summary
- **Dataset**: PlantVillage
- **Classes**: 15 target crop-disease classes
- **Total Samples**: 20,638 images
- **Partitioning**: 70% training, 15% validation, 15% testing

## Official Model Architectures
1. **MLP Baseline**: A non-spatial fully connected network.
2. **Custom CNN**: A 4-stage convolutional neural network trained from scratch.
3. **ResNet-18**: An 18-layer residual network fine-tuned from ImageNet-1K.
4. **ViT-B/16**: A vision transformer fine-tuned from ImageNet-1K.

## Additional Reference Benchmark
- **EfficientNet-B0**: A compound-scaled CNN used for additional reference.

## Common Experimental Protocol
All models were evaluated under identical conditions:
- Standardized image resolution of 224x224.
- Identical ImageNet normalization.
- Deterministic 70/15/15 dataset splits.
- Standardized computation of Top-1 Accuracy, Macro-Averaged Precision, Recall, Macro-F1, and Top-3 Accuracy.

## Current Implementation Status
All official models and the additional benchmark have been fully implemented, trained, and evaluated. A unified comparison framework has been successfully generated.

## Final Verified Preliminary-Result Table
| Model | Test Accuracy | Macro Precision | Macro Recall | Macro-F1 | Top-3 Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MLP** | 77.42% | 77.36% | 71.27% | 72.07% | 94.54% |
| **Custom CNN** | 76.94% | 78.84% | 71.76% | 70.81% | 91.63% |
| **ResNet-18** | 88.53% | 88.64% | 86.97% | 87.06% | 98.29% |
| **ViT-B/16** | 99.77% | 99.79% | 99.80% | 99.79% | 100.00% |

## Technical Work Completed
- Architecture implementations.
- Configuration-aware training loops.
- Dataset stratified splitting and loading.
- Official GPU execution and validation for ViT-B/16.
- Cross-model performance comparison script.

## Work Remaining for Final Submission
- Literature review (8 to 10 papers).
- Error analysis using confusion matrices.
- Final report drafting and formatting.
- Presentation slide creation.
