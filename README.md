# Visual-Question-Answering
Visual Question Answering (VQA) is the task of answering open-ended questions based on an image. VQA has many applications: Medical VQA, Education purposes, for surveillance and numerous other applications. 
Our objective is to develop an advanced system capable of responding to inquiries about images with the aim of helping visually impaired people, leveraging cutting-edge deep learning techniques and using [VizWiz dataset](https://www.kaggle.com/datasets/ingbiodanielh/vizwiz).


## Table of Contents

1. [Visual-Question-Answering](#visual-question-answering)
2. [Download the Dataset and Data Analysis](#download-the-dataset-and-data-analysis)
    - [VizWiz Dataset Overview](#vizwiz-dataset-overview)
    - [Dataset Link](#dataset-link)
    - [Test Data](#test-data)
    - [Data Analysis](#data-analysis)
    - [Data Mounting with Kaggle](#data-mounting-with-kaggle)
    - [Data Processing](#data-processing)
3. [Building the Model](#building-the-model)
    - [Implementation Reference](#implementation-reference)
    - [Utilize Pre-trained Models](#utilize-pre-trained-models)
    - [Model Training](#model-training)
    - [Code Modification](#code-modification)


# Download the Dataset and Data Analysis

### **VizWiz Dataset Overview:**
   - VizWiz is a Visual Question Answering (VQA) dataset containing 20,500 image/question pairs.
   - Each image is associated with a corresponding question and has 10 answers to that question.
     i. 20,523 training image/question pairs
     ii. 205,230 training answer/answer confidence pairs
     iii. 4,319 validation image/question pairs
     iv. 43,190 validation answer/answer confidence pairs

###**Dataset Link:**
   - The dataset can be found at the following link: [VizWiz Dataset](https://www.kaggle.com/datasets/ingbiodanielh/vizwiz)

### **Test Data:**
   - Take 5% of the training data as test data.

### **Data Analysis:**
   - Perform a comprehensive analysis of the data and present a comprehensible histogram.

### **Data Mounting with Kaggle:**
   - Utilize Kaggle to instantly mount the data without any hassle.

### **Data Processing:**
   - Set seed to 42 and use stratification while considering that answer/answer confidence pairs correspond to the images and questions.

# Building the Model

### **Implementation Reference:**
   - Follow the implementation details provided in the paper [2206.05281v1.pdf](https://arxiv.org/abs/2206.05281v1).

### **Utilize Pre-trained Models:**
   - Use the CLIP model's image encoder and text encoder provided by OpenAI. The relevant paper and GitHub repository can be found in [2103.00020.pdf](https://arxiv.org/abs/2103.00020).

### **Model Training:**
   - Train the classification layers following the paper's approach.
   - Concatenate resulting features from image and text encoders, pass them through linear layers with layer normalization, and apply a high dropout value (0.5).
   - Predict answer types and answers using an additional linear layer.
   - The visual encoder's image size is 448x448 for RN50x64 and 336x336 for ViT-L/14@336px.
   - Train the linear classifier using cross entropy loss with rotation as image augmentation.
   - Train only the additional linear classifier, keeping the CLIP model frozen. This allows fast and efficient training without large computational resources.

### **Code Modification:**
   - Modify the provided PyTorch code from the CLIP repository to implement this specific model.

