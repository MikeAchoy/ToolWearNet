
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

'''
    Code used to compare the ground truth mask to model
    predicted wear mask on tool.
'''

# Model and paths
model_path = "model/model.keras"
train_path = "model/stage3_train"
image_size = 128

# Load the trained model
model = keras.models.load_model(model_path, compile=False)


# Function to preprocess the image
def preprocess_image(image_path, image_size):
    image = cv2.imread(image_path, 1)
    image = cv2.resize(image, (image_size, image_size))
    image = image / 255.0
    return np.expand_dims(image, axis=0)  # Add batch dimension


# Function to post-process and display the mask
def display_prediction(image_num, model, train_path, image_size):
    # Construct paths
    image_id = os.listdir(train_path)[image_num]
    image_path = os.path.join(train_path, image_id, "images", f"{image_id}.jpg")

    # Preprocess the image
    image = preprocess_image(image_path, image_size)

    # Predict the mask
    predicted_mask = model.predict(image)
    predicted_mask = (predicted_mask[0] > 0.5).astype(np.uint8)  # Threshold and remove batch dimension

    # Read original image for display
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image = cv2.resize(original_image, (image_size, image_size))

    # Display the image and mask
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image)

    plt.subplot(1, 2, 2)
    plt.title("Predicted Mask")
    plt.imshow(predicted_mask, cmap="gray")

    plt.show()

# Predict and display the mask for a specific image number
# image_number = 0  # Change this to the desired image number
image_number = int(input("Enter image number: "))
display_prediction(image_number, model, train_path, image_size)
