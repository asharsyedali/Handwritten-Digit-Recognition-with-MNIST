# Handwritten Digit Recognition with MNIST

Welcome to the **Handwritten Digit Recognition with MNIST** project! This project demonstrates how to use a **Convolutional Neural Network (CNN)** to classify handwritten digits (0-9) using the **MNIST dataset**. The project includes both model training and an interactive web app for real-time digit prediction.

---

## 🚀 Features

- **Trained CNN Model**: A deep neural network trained on the MNIST dataset to classify handwritten digits.
- **Real-Time Prediction**: Draw a digit on the canvas, and the model will predict it instantly with a confidence score.
- **Interactive Web App**: Built with **Gradio**, allowing users to interact with the model through a simple and intuitive interface.
- **Confidence Scoring**: Displays the model's prediction confidence alongside the digit.

---

## 🧠 Technologies

The following technologies are used in this project:

- **TensorFlow / Keras**: Framework used for building and training the Convolutional Neural Network.
- **Gradio**: Library used to build a fast and easy-to-use web interface for the model.
- **NumPy**: Used for handling data manipulation and preprocessing.
- **OpenCV**: For image processing tasks such as resizing and normalizing the input images.
- **Matplotlib**: For visualizing training accuracy and loss over epochs.

---

## 📊 Dataset

This project uses the **MNIST dataset**, which contains 60,000 28x28 grayscale images of handwritten digits for training and 10,000 images for testing. It is widely used for benchmarking image classification models.

You can access the MNIST dataset [here](http://yann.lecun.com/exdb/mnist/).

---

## 🏗️ Model Architecture

The Convolutional Neural Network (CNN) used in this project consists of the following layers:

1. **Conv2D Layer** (32 filters, 3x3 kernel, ReLU activation): Extracts low-level features like edges.
2. **MaxPooling Layer** (2x2 pool size): Reduces the spatial dimensions of the image.
3. **Conv2D Layer** (64 filters, 3x3 kernel, ReLU activation): Detects more complex features.
4. **MaxPooling Layer** (2x2 pool size): Further reduces the image size.
5. **Flatten Layer**: Converts the 2D feature map into a 1D vector.
6. **Dropout Layer** (50%): Prevents overfitting by randomly disabling 50% of the neurons.
7. **Dense Layer**: 10 output units (one for each digit, 0-9) with a softmax activation function.

---

## 🚀 Usage

### 🖊️ Real-Time Web App:

The project includes an interactive web app where users can draw digits directly on a canvas, and the model will predict the digit in real-time.

#### Steps to use:
1. **Draw a digit** (0-9) on the canvas.
2. **Get the prediction**: The model will predict the digit and display the confidence score.
3. **Try different drawings**: The model will provide updated predictions instantly.

---

## 🛠️ Setup Instructions

### Prerequisites:
Make sure you have Python 3.x and the following libraries installed:

```bash
pip install -r requirements.txt
Training the Model:
Clone the repository:

bash
Copy
Edit
git clone https://github.com/<your-username>/Handwritten-Digit-Recognition-with-MNIST.git
cd Handwritten-Digit-Recognition-with-MNIST
Train the model:

bash
Copy
Edit
python train_model.py
This will train the CNN model on the MNIST dataset and generate a final_model.keras file.

Running the Real-Time App:
To launch the real-time app, run:

bash
Copy
Edit
python app.py
Gradio will provide a URL (e.g., https://<app-id>.gradio.app) that you can open in your browser to start drawing and predicting digits.

💻 Example Predictions
Here are some example predictions from the real-time app:


🌟 Contributing
Feel free to fork this repository and contribute! You can help by:

Improving the model's performance.
Adding additional features (e.g., support for custom datasets).
Fixing bugs or enhancing code performance.
Writing better documentation or adding more examples.
📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

👨‍💻 Author
This project was created by Syed Ali Ashar, as part of the University of Management and Technology (UMT) coursework.

🔗 Links
MNIST Dataset
Keras Documentation
Gradio Documentation
