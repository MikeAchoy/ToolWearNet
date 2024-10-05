import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

'''
    Attempt at combining mask prediction, and post processing in one file. 
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

def hough_transform(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (25, 25), 0)

    # Detect edges using Canny algorithm with the provided thresholds
    edges = cv2.Canny(blurred, 0, 100)

    # Apply Hough Transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)

    # Find the longest line
    longest_line = None
    max_length = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length > max_length:
                max_length = length
                longest_line = line

    # Draw the longest line on the original image
    if longest_line is not None:
        x1, y1, x2, y2 = longest_line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(image, f"Longest line: length = {max_length:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        XY.append(x1)
        XY.append(y1)
        XY.append(x2)
        XY.append(y2)

    # Display the result using matplotlib to enable zooming
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('on')  # Show axis for zooming
    plt.show()

def main(image_path):
    # Get the predicted mask
    predicted_mask = get_predicted_mask(image_path, model, image_size)

    # Debug: Display the predicted mask
    plt.imshow(predicted_mask, cmap='gray')
    plt.title("Predicted Mask")
    plt.show()

    # Perform Hough Transform on the image
    hough_transform(image_path)
    print(XY)

    # Use predicted mask for further processing
    color = cv2.merge([predicted_mask, predicted_mask, predicted_mask])
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=7)
    etalon = XY
    print("Etalon", etalon)

    # Create parallel segments
    def segments_paralleles(x1, y1, x2, y2, nb_segments, distance):
        segments = []
        dx = x2 - x1
        dy = y2 - y1
        norm = (dx**2 + dy**2)**0.5
        ux = dx / norm  # Vecteur unitaire dans la direction x
        uy = dy / norm  # Vecteur unitaire dans la direction y

        for i in range(- int(nb_segments), int(nb_segments)):
            offset_x = i * distance * uy  # Décalage dans la direction y
            offset_y = -i * distance * ux  # Décalage dans la direction x
            x1_new = x1 + offset_x
            y1_new = y1 + offset_y
            x2_new = x2 + offset_x
            y2_new = y2 + offset_y
            segments.append([int(x1_new),  int(y1_new), int(x2_new), int(y2_new)])

        return segments

    # Create line from segment
    def creer_droite(x1, y1, x2, y2, img_shape):
        if x2 != x1:
            denom = x2 - x1
        else:
            denom = np.inf
        a = (y2 - y1) / denom
        b = y1 - a * x1

        # Dessiner une ligne suffisamment longue pour simuler une droite infinie
        x3 = 0
        y3 = int(a * x3 + b)
        x4 = img_shape[1] - 1
        y4 = int(a * x4 + b)

        L = [x3, y3, x4, y4]
        return L

    # Find shape of the edges
    def point_du_contour(contour):
        Liste_contour = []
        for i in range(contour.shape[0]):
            for j in range(contour.shape[1]):
                if contour[i, j] != 0:
                    Liste_contour.append((j, i))
        return Liste_contour

    # Verification that the segment crosses the edges
    def appartient_contour(pt_contour, seg_para, epsilon):
        Les_restants = []
        sans_doublons = []

        for L in seg_para:
            x1, y1, x2, y2 = L
            if x2 != x1:
                denom = x2 - x1
            else:
                denom = np.inf
            a = (y2 - y1) / denom
            b = y1 - a * x1
            for pt in pt_contour:
                x, y = pt
                if abs(a * x + b - y) < epsilon:
                    Les_restants.append(L)

        for element in Les_restants:
            if element not in sans_doublons:
                sans_doublons.append(element)

        return sans_doublons

    # Find distance between 2 lines
    def distance_lignes(ligne1, ligne2):
        x1, y1, x2, y2 = ligne1
        x3, y3, x4, y4 = ligne2

        if x2 != x1:
            denom1 = x2 - x1
        else:
            denom1 = np.inf

        if x3 != x4:
            denom2 = x3 - x4
        else:
            denom2 = np.inf

        a1 = (y2 - y1) / denom1
        b1 = y1 - a1 * x1
        a2 = (y4 - y3) / denom2
        b2 = y3 - a2 * x3

        distance = abs(b2 - b1)
        return distance

    # Determine max distance
    def distance_maximale(ligne_etalon, lignes_para):
        distance_max = 0
        lignes_les_plus_eloignees = None

        for ligne in lignes_para:
            distance = distance_lignes(ligne_etalon, ligne)
            if distance > distance_max:
                distance_max = distance
                lignes_les_plus_eloignees = (ligne_etalon, ligne)

        return lignes_les_plus_eloignees, distance_max

    fichier = open("debug.txt", "a")

    # Get reference segment
    x1, y1 = XY[0], XY[1]
    x2, y2 = XY[2], XY[3]

    # Plot reference line
    I = creer_droite(x1, y1, x2, y2, color.shape)
    x3, y3 = I[0], I[1]
    x4, y4 = I[2], I[3]
    cv2.line(color, (x3, y3), (x4, y4), (0, 0, 200), 5)

    # Plot all the parallel lines
    distance = 1
    nb_segments = 100
    les_segments_paralleles = segments_paralleles(x1, y1, x2, y2, nb_segments, distance)
    for seg in les_segments_paralleles:
        xa, ya, xb, yb = seg

    # Get edges points
    Les_points_que_ils_appartiennent_a_le_contour = point_du_contour(edges)
    fichier.write(str(Les_points_que_ils_appartiennent_a_le_contour))

    # Get useful segment
    segments_utiles = appartient_contour(Les_points_que_ils_appartiennent_a_le_contour, les_segments_paralleles, 1)
    LE_segment, distance_max = distance_maximale(etalon, segments_utiles)

    if LE_segment is not None:
        xu1, yu1 = LE_segment[1][0], LE_segment[1][1]
        xu2, yu2 = LE_segment[1][2], LE_segment[1][3]

        # Print VB value
        dist_max_pix = distance_max
        dist_max_mm = dist_max_pix * 0.5 / 206
        print(f"Distance maximale: {dist_max_pix:8.3f} pixels")
        print(f"Distance maximale: {dist_max_mm:8.3f} mm")

        # Plot useful line
        D = creer_droite(xu1, yu1, xu2, yu2, color.shape)
        x4, y4 = D[0], D[1]
        x5, y5 = D[2], D[3]
        cv2.line(color, (x4, y4), (x5, y5), (0, 255, 50), 5)
    else:
        print("No useful segments found.")

    # Plot image with lines
    cv2.imshow('tete', edges)
    cv2.imshow("Image", color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    fichier.close()

# Define the image path that the model will use for wear mask prediction
image_number = input("Select dataset image number: ")
image_path = f"/Users/mikea./Desktop/ToolVBMeasurement/Tool Images/{image_number}.jpg"
XY = []

if __name__ == '__main__':
    main(image_path)
