# Risk and Remaining Plan

## Remaining Academic Work
- Complete and verify 8 to 10 paper literature-review table.
- Integrate citations into Related Work.
- Prepare report sections.
- Perform structured error analysis using confusion matrices and misclassified samples.
- Finalize model-comparison discussion.
- Create presentation slides.
- Prepare viva explanations.
- Prepare truthful individual-contribution statement.
- Produce plagiarism and AI-content reports according to course submission requirements.
- Final verification of GitHub repository and report consistency.

## Identified Risks
- **PlantVillage controlled-background bias**: The dataset images often feature uniform backgrounds and lab lighting, which may artificially inflate metrics compared to real-world scenarios.
- **Lack of external field-data evaluation**: No evaluation on noisy field datasets has been performed.
- **Single train/validation/test split**: Results are based on one fixed split, posing a risk of overfitting to the test set.
- **No repeated-seed statistical analysis**: Metrics lack variance margins.
- **Different pretraining status**: Comparisons confound architectural differences (CNN vs. Transformer) with pretraining effects (from-scratch vs. ImageNet-pretrained).
- **Limited ability to isolate architecture effects from pretraining effects**: E.g., ViT benefits massively from its ImageNet prior.
- **ViT computational requirements**: High training and inference overhead relative to lightweight CNNs.
