import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


'''
    Main model loading and prediction.
'''

# Model and paths
model_path = "model/model.keras"
image_size = 128

# Load the trained model
model = keras.models.load_model(model_path, compile=False)

# Function to preprocess the image
def preprocess_image(image_path, image_size):
    image = cv2.imread(image_path, 1)
    image = cv2.resize(image, (image_size, image_size))
    image = image / 255.0
    return np.expand_dims(image, axis=0)  # Add batch dimension

# Function to get the predicted mask
def get_predicted_mask(image_path, model, image_size):
    # Preprocess the image
    image = preprocess_image(image_path, image_size)

    # Predict the mask
    predicted_mask = model.predict(image)
    predicted_mask = (predicted_mask[0] > 0.5).astype(np.uint8)  # Threshold and remove batch dimension

    return predicted_mask

def main(image_path):
    # Get the predicted mask
    predicted_mask = get_predicted_mask(image_path, model, image_size)

    # Debug: Display the predicted mask
    plt.imshow(predicted_mask, cmap='gray')
    plt.title("Predicted Mask")
    plt.show()

    # Load the original image for display
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Display the original image
    plt.imshow(original_image)
    plt.title("Input Tool Image")
    plt.axis('off')  
    plt.show()

# Define the image path that the model will use for wear mask prediction.
# Change image_number to input image you want, or change image_path to the input image for model.
image_number = 4
image_path = f"Tool Images/{image_number}.jpg"

if __name__ == '__main__':
    main(image_path)
