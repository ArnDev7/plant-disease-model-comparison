# Literature Review

| No. | Paper | Year and Venue | Method / Architecture | Dataset | Key Verified Result | Relevance to This Project |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | Hughes et al., "An open access repository of images on plant health to enable the development of mobile disease diagnostics" | 2015, arXiv | Dataset Release | PlantVillage | Total Images: 54,306 (PlantVillage) | Establishes the PlantVillage dataset context used in this project. |
| 2 | Mohanty et al., "Using deep learning for image-based plant disease detection" | 2016, Frontiers in Plant Science | CNN (AlexNet/GoogLeNet) | PlantVillage | Accuracy: 99.35% (PlantVillage) | Serves as the primary early CNN baseline on the PlantVillage dataset. |
| 3 | Sladojevic et al., "Deep neural networks based recognition of plant diseases by leaf image classification" | 2016, Computational Intelligence and Neuroscience | Custom CNN | Own Dataset | Accuracy: 96.30% (Own Dataset) | Provides early historical context on CNN-based leaf disease recognition. |
| 4 | Ferentinos, "Deep learning models for plant disease detection and diagnosis" | 2018, Computers and Electronics in Agriculture | VGG/AlexNet | Open database | Accuracy: 99.53% (Open database) | Provides broad multi-species CNN context supporting the Custom CNN motivation. |
| 5 | Too et al., "A comparative study of fine-tuning deep learning models for plant disease identification" | 2019, Computers and Electronics in Agriculture | ResNet/VGG/Inception | PlantVillage | Accuracy: 99.00% (PlantVillage 38 classes) | Supports the use of transfer learning and ResNet for limited-domain agricultural imagery. |
| 6 | He et al., "Deep residual learning for image recognition" | 2016, CVPR | ResNet | ImageNet | Top-5 error: 3.57% (ImageNet) | Introduces residual learning, which underpins the project's ResNet-18 model. |
| 7 | Dosovitskiy et al., "An image is worth 16x16 words: Transformers for image recognition at scale" | 2021, ICLR | Vision Transformer (ViT) | ImageNet | Accuracy: 88.55% (ImageNet) | Introduces the ViT architecture used for the attention-based comparison. |
| 8 | Salman et al., "Plant disease classification in the wild using vision transformers and mixture of experts" | 2025, Frontiers in Plant Science | ViT + Mixture of Experts | PlantVillage / PlantDoc | Accuracy: 68.00% (Cross-domain) | Provides state-of-the-art context for using ViT on plant diseases. |
| 9 | Tan & Le, "EfficientNet: Rethinking model scaling for convolutional neural networks" | 2019, ICML | EfficientNet | ImageNet | Top-1 Accuracy: 84.30% (ImageNet) | Introduces the EfficientNet architecture used as an additional reference benchmark. |
| 10 | Saleem et al., "Plant disease detection and classification by deep learning" | 2019, Plants | Comparative Review | Various | N/A | Review paper framing the broad research gap and evolution of DL in agriculture. |

## Paper 1: An open access repository of images on plant health to enable the development of mobile disease diagnostics
- **Full citation**: D. P. Hughes and M. Salathé, "An open access repository of images on plant health to enable the development of mobile disease diagnostics," arXiv preprint arXiv:1511.08060, 2015.
- **Research problem**: The lack of large, publicly available, expertly curated datasets for training computer vision models to detect crop diseases.
- **Method**: Dataset curation and release.
- **Dataset**: PlantVillage.
- **Experimental setup**: Not applicable; the paper is a dataset release announcement detailing the collection of 54,306 images across 14 crops and 26 diseases.
- **Key result**: Successful compilation and open-access publication of the dataset.
- **Strength**: Unprecedented scale and public availability spurred rapid advancements in AI for agriculture.
- **Limitation**: Images were captured in highly controlled laboratory settings with uniform backgrounds, limiting real-world robustness.
- **Direct relevance to this project**: Establishes the PlantVillage dataset context used in this project.
- **Verified source link**: [https://arxiv.org/abs/1511.08060](https://arxiv.org/abs/1511.08060)
- **DOI link**: N/A
- **Verification notes**: Verified from arXiv metadata.

## Paper 2: Using deep learning for image-based plant disease detection
- **Full citation**: S. P. Mohanty, D. P. Hughes, and M. Salathé, "Using deep learning for image-based plant disease detection," Frontiers in Plant Science, vol. 7, p. 1419, 2016.
- **Research problem**: Automating the diagnosis of plant diseases using modern machine learning on smartphone-captured images.
- **Method**: Training deep Convolutional Neural Networks (AlexNet and GoogLeNet) using both from-scratch and transfer learning approaches.
- **Dataset**: PlantVillage.
- **Experimental setup**: 60 model configurations evaluated on 54,306 PlantVillage images using an 80/20 train/test split.
- **Key result**: The GoogLeNet model achieved an accuracy of 99.35% on a held-out test set.
- **Strength**: Demonstrated conclusively that transfer learning is highly effective for plant disease classification.
- **Limitation**: The model's accuracy dropped significantly (around 31%) when tested on images collected under different conditions (e.g., from the web).
- **Direct relevance to this project**: Serves as the primary early baseline for CNNs on the PlantVillage dataset.
- **Verified source link**: [https://www.frontiersin.org/articles/10.3389/fpls.2016.01419/full](https://www.frontiersin.org/articles/10.3389/fpls.2016.01419/full)
- **DOI link**: [10.3389/fpls.2016.01419](https://doi.org/10.3389/fpls.2016.01419)
- **Verification notes**: Verified from the official publisher (Frontiers).

## Paper 3: Deep neural networks based recognition of plant diseases by leaf image classification
- **Full citation**: S. Sladojevic, M. Arsenovic, A. Anderla, D. Culibrk, and D. Stefanovic, "Deep neural networks based recognition of plant diseases by leaf image classification," Computational Intelligence and Neuroscience, vol. 2016, 2016.
- **Research problem**: Developing an automated, image-based system to reliably distinguish between healthy leaves and 13 distinct plant diseases.
- **Method**: Designing and training a deep Convolutional Neural Network from scratch.
- **Dataset**: A custom dataset of 30,000 images downloaded from the Internet.
- **Experimental setup**: Preprocessing images to a fixed size, augmenting the data, and training a custom multi-layer CNN.
- **Key result**: Achieved an average accuracy of 96.30% across the 13 disease classes.
- **Strength**: Proved the viability of CNNs for this task without relying on existing large-scale structured datasets.
- **Limitation**: Limited transparency regarding the precise dataset splits and potential class overlaps.
- **Direct relevance to this project**: Provides early historical context on CNN-based leaf disease recognition.
- **Verified source link**: [https://www.hindawi.com/journals/cin/2016/3289801/](https://www.hindawi.com/journals/cin/2016/3289801/)
- **DOI link**: [10.1155/2016/3289801](https://doi.org/10.1155/2016/3289801)
- **Verification notes**: Verified from the official publisher (Hindawi).

## Paper 4: Deep learning models for plant disease detection and diagnosis
- **Full citation**: K. P. Ferentinos, "Deep learning models for plant disease detection and diagnosis," Computers and Electronics in Agriculture, vol. 145, pp. 311-318, 2018.
- **Research problem**: Expanding automated disease detection to a large number of plant-disease combinations using deep learning.
- **Method**: Benchmarking several prominent CNN architectures including AlexNet, AlexNetOWTBn, GoogLeNet, Overfeat, and VGG.
- **Dataset**: An open database comprising 87,848 images covering 58 distinct plant-disease combinations.
- **Experimental setup**: Training the aforementioned models using standardized deep learning practices on the large dataset.
- **Key result**: The VGG architecture achieved the highest classification accuracy of 99.53% on unseen test images.
- **Strength**: High accuracy across a very diverse set of crops and diseases.
- **Limitation**: Detailed methodological limitations could not be assessed from the accessible metadata or abstract.
- **Direct relevance to this project**: Provides broad multi-species CNN context supporting the Custom CNN motivation.
- **Verified source link**: [https://www.sciencedirect.com/science/article/pii/S016816991830030X](https://www.sciencedirect.com/science/article/pii/S016816991830030X)
- **DOI link**: [10.1016/j.compag.2018.01.009](https://doi.org/10.1016/j.compag.2018.01.009)
- **Verification notes**: Verified from the official publisher (Elsevier).

## Paper 5: A comparative study of fine-tuning deep learning models for plant disease identification
- **Full citation**: E. C. Too, L. Yujian, S. Njuki, and L. Yingchun, "A comparative study of fine-tuning deep learning models for plant disease identification," Computers and Electronics in Agriculture, vol. 161, pp. 272-279, 2019.
- **Research problem**: Identifying the most efficient and accurate deep learning architectures for agricultural imagery via transfer learning.
- **Method**: Comparing fine-tuned versions of VGG16, VGG19, InceptionV3, and ResNet50.
- **Dataset**: PlantVillage (subset of 38 classes).
- **Experimental setup**: Models pre-trained on ImageNet were fine-tuned using identical hyperparameters on the PlantVillage dataset.
- **Key result**: ResNet50 achieved a high accuracy of 99.00% while being computationally efficient compared to VGG models.
- **Strength**: Clearly demonstrated the architectural efficiency of residual networks over dense, sequential CNNs like VGG.
- **Limitation**: The study relies heavily on the clean PlantVillage data, lacking real-world field evaluation.
- **Direct relevance to this project**: Supports the use of transfer learning and ResNet for limited-domain agricultural imagery.
- **Verified source link**: [https://www.sciencedirect.com/science/article/pii/S0168169918302061](https://www.sciencedirect.com/science/article/pii/S0168169918302061)
- **DOI link**: [10.1016/j.compag.2018.03.032](https://doi.org/10.1016/j.compag.2018.03.032)
- **Verification notes**: Verified from the official publisher (Elsevier).

## Paper 6: Deep residual learning for image recognition
- **Full citation**: K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778, 2016.
- **Research problem**: Addressing the degradation problem where deeper neural networks exhibit higher training and test errors than shallower counterparts.
- **Method**: Introducing residual learning frameworks with shortcut connections that skip one or more layers.
- **Dataset**: ImageNet 2012 classification dataset.
- **Experimental setup**: Training exceptionally deep networks (up to 152 layers) and evaluating their error rates against plain networks.
- **Key result**: The 152-layer ResNet achieved a 3.57% top-5 error on the ImageNet test ensemble, winning the ILSVRC 2015 classification task.
- **Strength**: Elegantly solved the vanishing gradient and degradation problems, allowing networks to scale to hundreds of layers.
- **Limitation**: Residual blocks increase memory overhead during training due to the need to store activations for the skip connections.
- **Direct relevance to this project**: Introduces residual learning, which underpins the project's ResNet-18 model.
- **Verified source link**: [https://ieeexplore.ieee.org/document/7780459](https://ieeexplore.ieee.org/document/7780459)
- **DOI link**: [10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90)
- **Verification notes**: Verified from the official publisher (IEEE).

## Paper 7: An image is worth 16x16 words: Transformers for image recognition at scale
- **Full citation**: A. Dosovitskiy et al., "An image is worth 16x16 words: Transformers for image recognition at scale," ICLR, 2021.
- **Research problem**: Investigating whether the Transformer architecture, highly successful in NLP, can replace CNNs for computer vision tasks.
- **Method**: Treating images as sequences of flattened 16x16 2D patches and processing them with a standard Transformer encoder (Vision Transformer).
- **Dataset**: ImageNet, ImageNet-21k, and JFT-300M.
- **Experimental setup**: Pre-training the ViT models on large datasets and transferring them to smaller benchmarks like ImageNet and CIFAR-100.
- **Key result**: ViT-H/14 achieved 88.55% accuracy on ImageNet, outperforming state-of-the-art CNN baselines when pre-trained on sufficiently large data.
- **Strength**: Demonstrated that convolutions are not strictly necessary and that self-attention can capture global image context effectively.
- **Limitation**: ViT lacks the inductive biases of CNNs (translation equivariance and locality) and requires massive amounts of data to generalize well.
- **Direct relevance to this project**: Introduces the ViT architecture used for the attention-based comparison.
- **Verified source link**: [https://openreview.net/forum?id=YicbFdNTTy](https://openreview.net/forum?id=YicbFdNTTy)
- **DOI link**: N/A
- **Verification notes**: Verified from the official conference (ICLR/OpenReview).

## Paper 8: Plant disease classification in the wild using vision transformers and mixture of experts
- **Full citation**: Z. Salman, A. Muhammad, and D. Han, "Plant disease classification in the wild using vision transformers and mixture of experts," Frontiers in Plant Science, vol. 16, Article 1522985, 2025.
- **Research problem**: Mitigating the performance degradation of plant disease classification models when deployed in real-world agricultural environments with dynamic backgrounds and varying lighting.
- **Method**: Combining a Vision Transformer (ViT) backbone with a Mixture of Experts (MoE) module, utilizing a gating mechanism and entropy/orthogonal regularization.
- **Dataset**: PlantVillage and PlantDoc datasets.
- **Experimental setup**: Evaluating the model's cross-domain robustness by training on laboratory data and testing on "in-the-wild" datasets.
- **Key result**: The proposed model achieved 68% accuracy in a challenging cross-domain evaluation, a 20% improvement over the standard ViT baseline.
- **Strength**: Directly addresses the critical issue of model robustness and out-of-distribution generalization in agricultural AI.
- **Limitation**: The Mixture of Experts framework significantly increases computational complexity and inference time.
- **Direct relevance to this project**: Provides state-of-the-art context for using ViT on plant diseases.
- **Verified source link**: [https://www.frontiersin.org/articles/10.3389/fpls.2025.1522985/full](https://www.frontiersin.org/articles/10.3389/fpls.2025.1522985/full)
- **DOI link**: [10.3389/fpls.2025.1522985](https://doi.org/10.3389/fpls.2025.1522985)
- **Verification notes**: Verified from the official publisher (Frontiers).

## Paper 9: EfficientNet: Rethinking model scaling for convolutional neural networks
- **Full citation**: M. Tan and Q. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," in International conference on machine learning, pp. 6105-6114, 2019.
- **Research problem**: Finding a principled method to scale up Convolutional Neural Networks to achieve better accuracy without arbitrarily increasing depth, width, or resolution.
- **Method**: Introducing a compound scaling method that uniformly scales network width, depth, and resolution using a set of fixed scaling coefficients.
- **Dataset**: ImageNet.
- **Experimental setup**: Developing a baseline EfficientNet-B0 using neural architecture search and scaling it up to B7 to compare against existing architectures.
- **Key result**: EfficientNet-B7 achieved 84.3% top-1 accuracy on ImageNet while being 8.4x smaller and 6.1x faster on inference than the best existing CNN.
- **Strength**: Highly efficient scaling rules produced models that drastically reduced parameter counts while improving accuracy.
- **Limitation**: Detailed methodological limitations could not be assessed from the accessible metadata or abstract.
- **Direct relevance to this project**: Introduces the EfficientNet architecture used as an additional reference benchmark.
- **Verified source link**: [https://proceedings.mlr.press/v97/tan19a.html](https://proceedings.mlr.press/v97/tan19a.html)
- **DOI link**: N/A
- **Verification notes**: Verified from the official conference proceedings (PMLR).

## Paper 10: Plant disease detection and classification by deep learning
- **Full citation**: M. H. Saleem, J. Potgieter, and K. M. Arif, "Plant disease detection and classification by deep learning," Plants, vol. 8, no. 11, p. 468, 2019.
- **Research problem**: Synthesizing the fragmented literature on the application of deep learning in plant disease detection to identify trends, challenges, and future directions.
- **Method**: A comprehensive comparative literature review.
- **Dataset**: Various datasets covered across reviewed literature (including PlantVillage).
- **Experimental setup**: Not applicable; this is a review paper.
- **Key result**: Concluded that while deep learning yields high accuracy on controlled datasets, significant challenges remain regarding real-time detection, dataset bias, and field deployment.
- **Strength**: Provides a broad overview of methodologies and exposes the systemic gaps in evaluation protocols across the field.
- **Limitation**: As a review paper, it relies on the reported metrics of the primary sources without independent empirical verification.
- **Direct relevance to this project**: Review paper framing the broad research gap and evolution of DL in agriculture.
- **Verified source link**: [https://www.mdpi.com/2223-7747/8/11/468](https://www.mdpi.com/2223-7747/8/11/468)
- **DOI link**: [10.3390/plants8110468](https://doi.org/10.3390/plants8110468)
- **Verification notes**: Verified from the official publisher (MDPI).

# Literature Synthesis

## Evolution of Plant Disease Classification
The field of plant disease classification has transitioned dramatically from traditional machine learning techniques relying on handcrafted image features to end-to-end deep learning pipelines. Early foundational work, catalyzed by the release of large-scale open datasets like PlantVillage [1], enabled researchers to benchmark complex neural architectures. This catalyzed the widespread adoption of Convolutional Neural Networks (CNNs), which demonstrated state-of-the-art capability in extracting hierarchical visual features directly from raw leaf pixels [3, 4].

## CNN and Transfer-Learning Approaches
Following early successes with custom, from-scratch CNN architectures [3], researchers identified transfer learning as a highly effective mechanism to accelerate training and improve accuracy. Studies benchmarking architectures like AlexNet and GoogLeNet [2] proved that weights initialized from massive general-purpose datasets (e.g., ImageNet) could be successfully fine-tuned for agricultural domains, frequently achieving accuracies upwards of 99% on controlled subsets [4]. 

## Residual and Efficient Architectures
The introduction of residual connections [6] addressed the degradation problem inherent in very deep networks, leading to robust models like ResNet-18. Comparative studies on plant diseases confirmed that fine-tuning residual networks yields high parameter efficiency and predictive accuracy compared to older sequential architectures [5]. Parallel advancements in architecture search and compound scaling introduced models like EfficientNet [9], which serve as robust, lightweight benchmarks for real-world agricultural deployment.

## Attention and Vision Transformers
A recent paradigm shift in computer vision has seen the introduction of Vision Transformers (ViT) [7], which replace convolutions entirely with self-attention mechanisms. In the agricultural domain, ViTs have been shown to capture global leaf context effectively, often outperforming CNNs [8]. Recent literature explores augmenting these transformers with robust gating mechanisms to maintain performance when models transition from controlled lab environments to noisy field conditions [8].

## Research Gap Addressed by This Project
As highlighted by broad literature reviews [10], many studies report exceptionally high accuracies but lack consistent evaluation protocols, making cross-architectural comparisons difficult. This project addresses this gap by implementing a controlled, empirical comparison of dense, convolutional, residual, and transformer-based architectures. By enforcing a common PlantVillage split, shared preprocessing pipelines, and identical evaluation metrics, this study isolates architectural trade-offs between from-scratch baselines and pretrained models (ResNet-18, ViT-B/16). 

## Expected Research Questions
1. How do dense, convolutional, residual, and transformer-based architectures compare under a common PlantVillage evaluation protocol?
2. How do from-scratch baselines compare with ImageNet-pretrained architectures on the selected dataset?
3. How do accuracy and macro-F1 rankings differ across the models?
4. Which disease classes show the largest performance differences across model families?

# References
[1] D. P. Hughes and M. Salathé, "An open access repository of images on plant health to enable the development of mobile disease diagnostics," *arXiv preprint arXiv:1511.08060*, 2015. [Source](https://arxiv.org/abs/1511.08060)

[2] S. P. Mohanty, D. P. Hughes, and M. Salathé, "Using deep learning for image-based plant disease detection," *Frontiers in Plant Science*, vol. 7, p. 1419, 2016. DOI: [10.3389/fpls.2016.01419](https://doi.org/10.3389/fpls.2016.01419). [Source](https://www.frontiersin.org/articles/10.3389/fpls.2016.01419/full)

[3] S. Sladojevic, M. Arsenovic, A. Anderla, D. Culibrk, and D. Stefanovic, "Deep neural networks based recognition of plant diseases by leaf image classification," *Computational Intelligence and Neuroscience*, vol. 2016, 2016. DOI: [10.1155/2016/3289801](https://doi.org/10.1155/2016/3289801). [Source](https://www.hindawi.com/journals/cin/2016/3289801/)

[4] K. P. Ferentinos, "Deep learning models for plant disease detection and diagnosis," *Computers and Electronics in Agriculture*, vol. 145, pp. 311-318, 2018. DOI: [10.1016/j.compag.2018.01.009](https://doi.org/10.1016/j.compag.2018.01.009). [Source](https://www.sciencedirect.com/science/article/pii/S016816991830030X)

[5] E. C. Too, L. Yujian, S. Njuki, and L. Yingchun, "A comparative study of fine-tuning deep learning models for plant disease identification," *Computers and Electronics in Agriculture*, vol. 161, pp. 272-279, 2019. DOI: [10.1016/j.compag.2018.03.032](https://doi.org/10.1016/j.compag.2018.03.032). [Source](https://www.sciencedirect.com/science/article/pii/S0168169918302061)

[6] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 770-778, 2016. DOI: [10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90). [Source](https://ieeexplore.ieee.org/document/7780459)

[7] A. Dosovitskiy et al., "An image is worth 16x16 words: Transformers for image recognition at scale," *ICLR*, 2021. [Source](https://openreview.net/forum?id=YicbFdNTTy)

[8] Z. Salman, A. Muhammad, and D. Han, "Plant disease classification in the wild using vision transformers and mixture of experts," *Frontiers in Plant Science*, vol. 16, Article 1522985, 2025. DOI: [10.3389/fpls.2025.1522985](https://doi.org/10.3389/fpls.2025.1522985). [Source](https://www.frontiersin.org/articles/10.3389/fpls.2025.1522985/full)

[9] M. Tan and Q. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," in *International conference on machine learning*, pp. 6105-6114, 2019. [Source](https://proceedings.mlr.press/v97/tan19a.html)

[10] M. H. Saleem, J. Potgieter, and K. M. Arif, "Plant disease detection and classification by deep learning," *Plants*, vol. 8, no. 11, p. 468, 2019. DOI: [10.3390/plants8110468](https://doi.org/10.3390/plants8110468). [Source](https://www.mdpi.com/2223-7747/8/11/468)

*AI-assisted tools were used to organize and refine the literature-review structure. Paper metadata, reported metrics, and source links must remain traceable to the verified academic sources listed above.*
