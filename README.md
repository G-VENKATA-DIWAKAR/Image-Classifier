Image Classifier:
This project implements a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset. It provides a user-friendly interface for image classification using Gradio.

Features
*CNN Model: Utilizes a CNN architecture to classify images into 10 classes.
*Gradio Interface: Offers a simple and intuitive web interface for users to upload images and receive classification results in real-time.
*Image Preprocessing: Ensures that input images are properly resized before being fed into the 
model.
*Callbacks: Implements EarlyStopping and ModelCheckpoint callbacks during model training for efficient training and model saving.
Usage:
1.Clone the repository:
git clone https://github.com/your-username/cifar-10-image-classifier.git
2.Install the required dependencies:
pip install -r requirements.txt
3.Run the script:
python cifar_10_image_classifier.py
4.Open your web browser and go to the provided URL to access the Gradio interface.

Dependencies:
*TensorFlow
*Gradio
*NumPy
Contributing:
Contributions are welcome! Please feel free to open a pull request or submit an issue if you encounter any bugs or have suggestions for improvements.
