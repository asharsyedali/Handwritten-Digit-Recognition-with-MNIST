# Handwritten Digit Recognition with MNIST

This project is a machine learning application that uses a **Convolutional Neural Network (CNN)** to recognize handwritten digits (0-9) from the **MNIST dataset**. The project includes both the model training pipeline and an interactive web app, allowing users to draw digits on a canvas and get instant predictions from the trained model.

## Table of Contents
- [Features](#features)
- [Technologies](#technologies)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [Setup Instructions](#setup-instructions)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

## Features
- **Trained CNN Model**: A deep neural network trained on the MNIST dataset to recognize handwritten digits.
- **Real-Time Prediction**: Users can draw digits directly on the canvas, and the model will predict the digit in real-time.
- **Confidence Scores**: Displays the model's prediction confidence for each drawn digit.
- **Interactive Web App**: Built with Gradio to create a simple user interface for real-time predictions.
- **Model Export**: The trained model can be used in various environments for deployment or further development.

## Technologies
- **TensorFlow / Keras**: Framework used to build and train the Convolutional Neural Network.
- **Gradio**: Library used to create a fast and interactive web app.
- **NumPy**: For data manipulation and preprocessing.
- **OpenCV**: For image processing tasks like resizing and normalizing the images.
- **Matplotlib**: For visualizing training accuracy and loss.

## Dataset
The **MNIST dataset** consists of 60,000 28x28 grayscale images of handwritten digits for training and 10,000 images for testing. It is one of the most widely used datasets in machine learning and is commonly used for benchmarking image classification models.

You can download the MNIST dataset directly from [here](http://yann.lecun.com/exdb/mnist/).

## Model Architecture
The CNN model used in this project consists of the following layers:
- **Convolutional Layer (32 filters, 3x3 kernel, ReLU activation)**: Extracts basic features such as edges and textures.
- **MaxPooling Layer (2x2 pool size)**: Reduces the spatial dimensions to prevent overfitting and make the model more efficient.
- **Convolutional Layer (64 filters, 3x3 kernel, ReLU activation)**: Detects more complex patterns.
- **MaxPooling Layer (2x2 pool size)**: Further reduces spatial dimensions.
- **Flatten Layer**: Converts the 2D feature maps into a 1D vector.
- **Dropout Layer (50%)**: Prevents overfitting by randomly dropping half of the neurons during training.
- **Fully Connected Layer**: 10 output units (one for each digit, 0-9) with softmax activation for classification.

## Usage
### Real-Time Web App:
The project includes an interactive web app built using **Gradio**, where users can draw a digit and get predictions. The app will show the predicted digit along with a confidence score.

### Steps:
1. Draw a digit (0-9) in the canvas.
2. The app will predict the digit and show the confidence score.
3. You can experiment with different drawings, and the model will provide updated predictions in real-time.

## Setup Instructions
### Prerequisites:
Make sure you have Python 3.x and the following libraries installed:

- **TensorFlow** (for model training)
- **Gradio** (for the interactive web app)
- **NumPy** (for data handling)
- **OpenCV** (for image processing)
- **Matplotlib** (for plotting training results)

To install the required libraries, run:

```bash
pip install -r requirements.txt
