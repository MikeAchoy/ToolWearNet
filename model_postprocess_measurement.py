import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

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

# Function to get the predicted mask
def get_predicted_mask(image_num, model, train_path, image_size):
    # Construct paths
    image_id = os.listdir(train_path)[image_num]
    image_path = os.path.join(train_path, image_id, "images", f"{image_id}.jpg")

    # Preprocess the image
    image = preprocess_image(image_path, image_size)

    # Predict the mask
    predicted_mask = model.predict(image)
    predicted_mask = (predicted_mask[0] > 0.5).astype(np.uint8)  # Threshold and remove batch dimension

    return predicted_mask, image_path

def calculate_vbmax(mask):
    # Apply Canny edge detection to the predicted mask
    edges = cv2.Canny(mask, 50, 150, apertureSize=7)

    # Debug: Display edges detected by Canny
    plt.imshow(edges, cmap='gray')
    plt.title("Edges detected by Canny on Predicted Mask")
    plt.show()

    # Find the maximum distance between white pixels on the y-axis
    y_indices, x_indices = np.where(edges == 255)
    
    if len(y_indices) == 0 or len(x_indices) == 0:
        print("No edges detected in the mask.")
        return None

    # Find the maximum distance between points on the y-axis
    max_y_distance = max(y_indices) - min(y_indices)

    # Convert pixel distance to mm (assuming a fixed conversion factor)
    dist_max_mm = max_y_distance * 0.5 / 206
    return dist_max_mm

def main(image_number):
    # Get the predicted mask
    predicted_mask, original_image_path = get_predicted_mask(image_number, model, train_path, image_size)

    # Debug: Display the predicted mask
    plt.imshow(predicted_mask, cmap='gray')
    plt.title("Predicted Mask")
    plt.show()

    # Calculate VBmax
    VBmax = calculate_vbmax(predicted_mask)
    if VBmax is not None:
        print(f"VBmax: {VBmax:.3f} mm")

    # Load the original image for display
    original_image = cv2.imread(original_image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Display the original image with detected edges overlaid
    plt.imshow(original_image)
    plt.axis('off')  # Hide axis for display
    plt.show()

# Predict and process the image
image_number = int(input("Enter image number: "))
main(image_number)
