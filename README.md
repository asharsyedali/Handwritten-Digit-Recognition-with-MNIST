# Handwritten Digit Recognition with MNIST

This project is a machine learning application that recognizes handwritten digits (0-9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset. The project also features a real-time web app where users can draw digits and get predictions instantly.

## Features
- **CNN Model**: Built with Keras and trained on the MNIST dataset.
- **Real-Time Prediction App**: Users can draw digits directly on a web-based canvas to get predictions.
- **User-Friendly Interface**: Powered by Gradio for ease of use.

## Project Contents
- `train_model.py`: Python script to train the CNN model on the MNIST dataset.
- `app.py`: Python script for the real-time digit recognition web app.
- `final_model.keras`: Pre-trained Keras model file.
- `example_images/`: Folder containing sample digit images.

## Installation
To set up the project locally, follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/asharsyedali/Handwritten-Digit-Recognition-with-MNIST.git
   cd Handwritten-Digit-Recognition-with-MNIST
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the real-time app:

bash
Copy
Edit
python app.py
The app will launch, and you’ll get a link to access it.

Usage
Draw a digit (0-9) on the canvas provided in the app.
The model will predict the digit and display the confidence score.
Dataset
The MNIST dataset contains 60,000 training images and 10,000 test images of handwritten digits, each of size 28x28 pixels.

Model Architecture
The CNN consists of:

Two convolutional layers with ReLU activation.
Two max-pooling layers.
Dropout for regularization.
Fully connected layer with softmax for output.
Example
Here’s an example prediction from the app:


Author
This project was created by Syed Ali Ashar as part of UMT coursework. Feel free to contribute or use this project for educational purposes.
