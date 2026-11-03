# Model Methodology

### MLP
- **Flattened image input**: Processes flattened raw pixel vectors (150,528 features for 224x224x3).
- **Verified layer structure**: Multi-layer perceptron.
- **Activation functions**: ReLU.
- **Batch normalization**: Included to stabilize training.
- **Dropout**: Used for regularization.
- **Output layer**: 15-class linear projection.
- **Training from scratch**: Yes.
- **Academic role as a non-spatial baseline**: Establishes the performance floor for non-spatial architectures.

### Custom CNN
- **Each convolution block**: 4 stages of convolution.
- **Channel progression**: Expanding channel dimensions across layers.
- **Batch normalization**: Present after convolutions.
- **ReLU**: Activation function used.
- **Max pooling**: Used for spatial downsampling.
- **Adaptive average pooling**: Global average pooling before the classifier.
- **Classifier**: Linear layer mapping to 15 classes.
- **Training from scratch**: Yes.
- **Academic role as the convolutional baseline**: Provides a baseline for models incorporating spatial inductive biases without pretraining.

### ResNet-18
- **torchvision architecture**: ResNet-18 from `torchvision.models`.
- **ImageNet pretraining**: Initialized with `ResNet18_Weights.IMAGENET1K_V1`.
- **Replaced classifier**: Final fully connected layer replaced for 15 classes.
- **Verified freeze/fine-tuning approach**: Two-phase training. Backbone frozen initially while the head is trained, then the entire network is fine-tuned.
- **Residual architecture role**: Evaluates the impact of residual connections and transfer learning.

### ViT-B/16
- **torchvision architecture**: ViT-B/16 from `torchvision.models`.
- **ViT_B_16_Weights.DEFAULT**: Initialized with default ImageNet weights.
- **Patch size**: 16x16 pixels.
- **Patch count**: 196 image patches (for 224x224 input).
- **Embedding dimension**: 768.
- **Replaced head**: MLP head replaced for 15 classes.
- **Head-only phase**: Initial phase training only the new classification head.
- **Fine-tuning phase**: Subsequent phase fine-tuning the entire transformer.
- **GPU execution**: Verified and executed on CUDA.
- **Attention-based architecture role**: Assesses the performance of self-attention mechanisms without convolutions.

### EfficientNet-B0 (Additional Benchmark)
- Included as an additional reference benchmark for a compound-scaled convolutional network.
