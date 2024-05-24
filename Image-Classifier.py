pip install gradio

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import gradio as gr


(X_train, Y_train), (X_test, Y_test) = datasets.cifar10.load_data()


X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0


classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])


cnn.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])


cnn.fit(X_train, Y_train, epochs=10, validation_data=(X_test, Y_test))


def predict_image(image):
    
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)

    
    y_pred = cnn.predict(image)[0]


    predicted_class = classes[np.argmax(y_pred)]

    return predicted_class


interface = gr.Interface(
    fn=predict_image,
    inputs=gr.inputs.Image(shape=(32, 32)),
    outputs=gr.outputs.Textbox(),
    live=True
)


interface.launch()
