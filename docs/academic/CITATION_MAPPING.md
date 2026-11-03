# Citation Mapping

| Reference | Introduction | Related Work | Dataset | Methodology | Discussion | Reason |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| [1] Hughes & Salathé (2015) | Yes | Yes | Yes | No | No | Establishes the PlantVillage dataset origin and core motivation. |
| [2] Mohanty et al. (2016) | Yes | Yes | No | No | Yes | Serves as the primary early baseline for CNNs on PlantVillage. |
| [3] Sladojevic et al. (2016) | Yes | Yes | No | No | No | Provides early historical context on CNN-based leaf disease recognition. |
| [4] Ferentinos (2018) | No | Yes | No | No | Yes | Broad multi-species CNN context supporting the Custom CNN motivation. |
| [5] Too et al. (2019) | No | Yes | No | No | Yes | Directly supports the use of fine-tuning and ResNet for PlantVillage. |
| [6] He et al. (2016) | No | No | No | Yes | No | Foundational citation for the ResNet-18 methodology section. |
| [7] Dosovitskiy et al. (2021) | No | No | No | Yes | No | Foundational citation for the ViT-B/16 methodology section. |
| [8] Salman et al. (2025) | No | Yes | No | No | Yes | Provides state-of-the-art context for using ViT on plant diseases. |
| [9] Tan & Le (2019) | No | No | No | Yes | No | Foundational citation for the EfficientNet-B0 methodology section. |
| [10] Saleem et al. (2019) | Yes | Yes | No | No | Yes | Review paper framing the broad research gap and evolution. |

## Suggested Related Work Paragraph Plan

**Paragraph 1: PlantVillage and early deep-learning work**
Discuss the release of the PlantVillage dataset [1] and the initial shift toward deep convolutional models for agricultural diagnosis, prominently highlighted by early CNN baselines [2, 3].

**Paragraph 2: Custom CNN and transfer-learning approaches**
Review the establishment of broad CNN pipelines [4] and how the field transitioned from training architectures from scratch to utilizing transfer learning to overcome limited domain data.

**Paragraph 3: Residual and efficient CNN architectures**
Examine how residual networks and deeper convolutional architectures improved classification performance on plant disease datasets through fine-tuning techniques [5], building upon their foundational success in general image recognition [6, 9].

**Paragraph 4: Attention and transformer-based approaches**
Discuss the recent paradigm shift toward Vision Transformers [7], highlighting how self-attention mechanisms are being applied to capture global leaf context and improve robustness in plant disease detection [8].

**Paragraph 5: Gap and positioning of the present comparative study**
Synthesize the reviews [10] to highlight the lack of consistent evaluation protocols. Position this project as a controlled, multi-family architectural comparison (Dense, CNN, ResNet, ViT) using identical data splits, metrics, and preprocessing to isolate architectural trade-offs.
