# classifyPbreedsModel
# Deep Learning Project
This repository contains the code for a Deep Learning project. The project aims to classify pet breeds based on images using various deep learning models. The code is written in Python using TensorFlow, a popular deep learning framework.

# Prerequisites
Before running the code, make sure you have the following libraries installed:

TensorFlow
Matplotlib
Pandas
NumPy
scikit-learn
# Dataset
The project uses a dataset of pet breed images, which can be downloaded from Kaggle. To download the dataset, the kaggle command-line tool is used. Make sure you have the tool installed and configured with your Kaggle API credentials.

# Data Preprocessing
The code includes a function to create a dataframe that holds the paths to the images and their corresponding labels. The images are read, decoded, and resized using TensorFlow functions. Data augmentation techniques like random flipping, brightness adjustment, and saturation adjustment are also applied to enhance the dataset.

Train-Test Split
The dataset is split into training and testing sets using the train_test_split function from scikit-learn. The split is stratified based on the labels to ensure balanced representation in both sets.

One-Hot Encoding
The target labels are converted into one-hot vectors using the tf.one_hot function from TensorFlow.

# Data Pipeline
A data pipeline is created using TensorFlow's tf.data.Dataset API. The pipeline shuffles the data, applies preprocessing functions, batches the data, and prefetches it for efficient processing during training.

# Image Visualization
The code includes a visualization of augmented images from the training dataset. Several images are randomly selected and plotted using Matplotlib.

Modeling
The project explores various deep learning models, with a focus on the ResNet architecture. Multiple Transfer Learning models are used to leverage pre-trained weights for improved performance. Evaluation metrics are used to choose the best model for the classification task.

CNN Model
