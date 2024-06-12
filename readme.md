
# Tomato Leaf Disease Detection using Deep Learning

## Project Overview

This project aims to detect diseases in tomato leaves using Convolutional Neural Networks (CNNs), specifically leveraging transfer learning with ResNet-50 and AlexNet models. The goal is to provide an efficient and accurate method for identifying common diseases affecting tomato crops, crucial for improving agricultural productivity.

## Table of Contents

- [Abstract](#abstract)
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Data Augmentation](#data-augmentation)
- [Models Used](#models-used)
- [Explainability](#explainability)
- [Results](#results)
- [Conclusion](#conclusion)
- [Usage](#usage)
- [References](#references)

## Abstract

The project outlines a method for detecting diseases in tomato foliage using CNNs. Transfer learning is employed by importing a pretrained ResNet-50 model and tailoring it to specific classification needs. Data augmentation strategies are applied to improve model performance. The system, developed using the PyTorch framework, achieves an accuracy of 91.63%. Additionally, the AlexNet model was found to outperform ResNet-50 in this task. An explainability method called Shapley has also been implemented for both models.

## Introduction

Tomatoes are a staple crop in Indian agriculture, making timely disease detection crucial. Traditional methods like visual inspection are often inaccurate and slow. This project uses deep learning, which has proven more effective in handling large datasets without extensive preprocessing. The project employs ResNet-50 and AlexNet models to classify six common tomato leaf diseases.

## Dataset

The dataset comprises 9,801 images of tomato leaves, including both healthy and diseased samples. The images were sourced from the PlantVillage repository and divided into 70% for training and 30% for testing.

## Data Augmentation

Data augmentation is critical in this project to improve feature extraction and reduce overfitting. Techniques such as scaling, rotation, and resizing were applied, increasing the dataset size to 39,204 images.

## Models Used

### ResNet-50

ResNet-50, a deep neural network with 50 layers, is used for its ability to learn complex features from images. Transfer learning allows the model to be fine-tuned for specific classification tasks.

### AlexNet

AlexNet is another deep learning model used for comparison. It processes input images through a series of convolutional and fully connected layers, applying techniques like ReLU activation and dropout to improve performance and reduce overfitting.

## Explainability

The Shapley value method from cooperative game theory has been adapted to provide insights into the models' predictions. This explainability approach helps in understanding the contribution of each feature to the model's decisions.

## Results

- **Accuracy**: AlexNet achieved an accuracy of 94.6%, outperforming ResNet-50, which achieved 91.6%.
- **ROC Curve**: Both models showed high ROC AUC values, indicating excellent classification performance.
- **Confusion Matrix**: Detailed performance metrics across different disease categories were evaluated.

## Conclusion

A robust tomato leaf disease detection model was developed using PyTorch. The AlexNet model demonstrated superior performance compared to ResNet-50. The addition of explainability techniques like Shapley values provided valuable insights into the models' decision-making processes.

## Usage

To use the models, follow these steps:

1. **Clone the repository**:
    ```sh
    git clone https://github.com/subash-j-suresh/Plant-Leaf-Disease-Detection.git
    cd tomato-leaf-disease-detection
    ```

2. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Run the training script**:
    ```sh
    python train.py
    ```

4. **Evaluate the model**:
    ```sh
    python evaluate.py
    ```

5. **Run inference on new data**:
    ```sh
    python predict.py --image_path path_to_image
    ```

## References

- PlantVillage Dataset: [Link](https://plantvillage.psu.edu)
- PyTorch Framework: [Link](https://pytorch.org)
- ResNet-50: He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
- AlexNet: Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems (pp. 1097-1105).
