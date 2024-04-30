# Vision-Transformer-vs-CNN
Comparison of Convolutional Neural Networks and Vision Transformers (ViTs)

![vision-transformer](https://github.com/ehzalp/Vision-Transformer-vs-CNN/assets/80691995/a2bf6852-4606-4a0e-93af-038ba99cd699)


  CNNs have been the de facto standard for visual recognition tasks for the better part of a decade. Yet, recent developments in the world of deep learning have introduced a new contender in the realm of computer vision: the Vision Transformer (ViT).
  Originally designed for natural language processing tasks, Transformer architectures have proven their versatility and effectiveness across multiple domains. In 2020, researchers adapted the Transformer model for image recognition tasks, and the Vision Transformer was born. 
As ViTs continue to gain traction, a pressing question arises: Will Vision Transformers replace CNNs as the go-to architecture for computer vision applications?


Summary
  Note that the state of the art results reported in the paper are achieved by pre-training the ViT model using the JFT-300M dataset, then fine-tuning it on the target dataset. To improve the model quality without pre-training, you can try to train the model for more epochs, use a larger number of Transformer layers, resize the input images, change the patch size, or increase the projection dimensions. Besides, as mentioned in the paper, the quality of the model is affected not only by architecture choices, but also by parameters such as the learning rate schedule, optimizer, weight decay, etc. In practice, it's recommended to fine-tune a ViT model that was pre-trained using a large, high-resolution dataset

  The Vision Transformer (ViT) is a model architecture that adapts the transformer framework—originally developed for natural language processing (NLP)—to computer vision tasks. It was introduced in a paper titled "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" by researchers at Google Brain. Here's an overview of how the Vision Transformer architecture works:

### 1. **Input Image and Patching**
   - **Image Patching**: The input image is first divided into fixed-size patches. For example, an image might be split into patches of 16x16 pixels each.
   - **Flatten and Linear Embedding**: These patches are then flattened into vectors and linearly projected (using a learned linear transformation). This transformation is akin to token embedding in NLP.

### 2. **Positional Encoding**
   - **Add Positional Information**: Similar to how positional encodings are added in NLP transformers to provide information about the sequence order, positional embeddings are added to the patch embeddings to retain positional information of the patches in the image.

### 3. **Transformer Encoder**
   - **Stack of Transformer Layers**: The core of the Vision Transformer is a series of transformer layers. Each layer has two main components:
     - **Multi-head Self-Attention**: This mechanism allows the model to weigh the importance of different patches relative to each other. It is key in enabling the model to focus on important parts of the image.
     - **Feed-Forward Neural Network**: This is applied independently to each patch position.

### 4. **Classification Head**
   - **Pooling**: After the final transformer block, the output corresponding to the first patch (often treated as a "class token" similar to the [CLS] token in BERT) is used to represent the entire image.
   - **Linear Layer**: This token is passed through a final linear layer to produce the output logits for classification.

### 5. **Training and Scaling**
   - **Pre-training and Fine-tuning**: ViTs are often pre-trained on large datasets using self-supervised or supervised learning, and then fine-tuned on specific tasks.
   - **Scaling**: The model's performance generally improves with larger model sizes and more training data, following the trend observed in NLP transformers.

### Advantages and Challenges
- **Advantages**:
  - ViT can achieve state-of-the-art performance on image classification tasks, especially when trained with enough data.
  - The self-attention mechanism allows it to focus on relevant parts of the image, potentially leading to better interpretability.

- **Challenges**:
  - Requires large amounts of data to train effectively from scratch.
  - High computational cost due to the self-attention mechanism, especially for larger images and more patches.

The introduction of Vision Transformers marked a significant shift in computer vision, moving away from traditional convolutional neural networks (CNNs) towards architectures that leverage the scalability and flexibility of transformers.

# RESOURCES 
Dataset : https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
Offical Paper : An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale - https://arxiv.org/abs/2010.11929


