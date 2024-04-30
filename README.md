# Vision-Transformer-vs-CNN
Comparison of Convolutional Neural Networks and Vision Transformers (ViTs)
[Uploading brainfmri-vitvscnn.ipynb…]()

![vision-transformer](https://github.com/ehzalp/Vision-Transformer-vs-CNN/assets/80691995/a2bf6852-4606-4a0e-93af-038ba99cd699)


  CNNs have been the de facto standard for visual recognition tasks for the better part of a decade. Yet, recent developments in the world of deep learning have introduced a new contender in the realm of computer vision: the Vision Transformer (ViT).
  Originally designed for natural language processing tasks, Transformer architectures have proven their versatility and effectiveness across multiple domains. In 2020, researchers adapted the Transformer model for image recognition tasks, and the Vision Transformer was born. 
As ViTs continue to gain traction, a pressing question arises: Will Vision Transformers replace CNNs as the go-to architecture for computer vision applications?


Summary
  Note that the state of the art results reported in the paper are achieved by pre-training the ViT model using the JFT-300M dataset, then fine-tuning it on the target dataset. To improve the model quality without pre-training, you can try to train the model for more epochs, use a larger number of Transformer layers, resize the input images, change the patch size, or increase the projection dimensions. Besides, as mentioned in the paper, the quality of the model is affected not only by architecture choices, but also by parameters such as the learning rate schedule, optimizer, weight decay, etc. In practice, it's recommended to fine-tune a ViT model that was pre-trained using a large, high-resolution dataset

# RESOURCES 
Offical Paper : An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale - https://arxiv.org/abs/2010.11929


