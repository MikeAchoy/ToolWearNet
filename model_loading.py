import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

'''
    PJT Student approach
'''

class AugmentedDataGen(keras.utils.Sequence):
    def __init__(self, ids, path, batch_size=8, image_size=128, image_datagen=None):
        self.ids = ids
        self.path = path
        self.batch_size = batch_size
        self.image_size = image_size
        self.image_datagen = image_datagen  # Ajout de l'attribut image_datagen
        self.on_epoch_end()

    def __load__(self, id_name):
        # Chemin
        image_path = os.path.join(self.path, id_name, "images", id_name) + ".jpg"
        mask_path = os.path.join(self.path, id_name, "masks/")
        all_masks = os.listdir(mask_path)

        # Lecture de l'image
        image = cv2.imread(image_path, 1)
        image = cv2.resize(image, (self.image_size, self.image_size))

        mask = np.zeros((self.image_size, self.image_size, 1))  # Segmentation binaire

        # Lecture des masques
        for name in all_masks:
            _mask_path = mask_path + name
            _mask_image = cv2.imread(_mask_path, -1)
            _mask_image = cv2.resize(_mask_image, (self.image_size, self.image_size))
            _mask_image = np.expand_dims(_mask_image, axis =-1)
            mask = np.maximum(mask, _mask_image)

        # Normalisation
        image = image / 255.0
        mask = mask / 255.0

        return image, mask


    def __getitem__(self, index):
        if (index+1)*self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index*self.batch_size

        files_batch = self.ids[index*self.batch_size : (index+1)*self.batch_size]

        image = []
        mask = []

        for id_name in files_batch:
            _img, _mask = self.__load__(id_name)
            image.append(_img)
            mask.append(_mask)

        image = np.array(image)
        mask = np.array(mask)

        if self.image_datagen:
            # Génération de données augmentées
            image = self.image_datagen.flow(image, batch_size=self.batch_size, shuffle=False).next()
        return image, mask


    def on_epoch_end(self):
        np.random.shuffle(self.ids)


    def __len__(self):
        return int(np.ceil(len(self.ids) / float(self.batch_size)))


# Define the convolutional block
def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c


# Define the down-sampling block
def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = conv_block(x, filters, kernel_size, padding, strides)
    p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
    return c, p


# Define the up-sampling block
def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    us = keras.layers.UpSampling2D((2, 2))(x)
    concat = keras.layers.Concatenate()([us, skip])
    c = conv_block(concat, filters, kernel_size, padding, strides)
    return c

# Define the bottleneck block
def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    return conv_block(x, filters, kernel_size, padding, strides)


image_size = 128


# Define the UNet model architecture
def UNet():
    f = [16, 32, 64, 128, 256]
    inputs = keras.layers.Input((image_size, image_size, 3))

    p0 = inputs
    c1, p1 = down_block(p0, f[0])  # 128 -> 64
    c2, p2 = down_block(p1, f[1])  # 64 -> 32
    c3, p3 = down_block(p2, f[2])  # 32 -> 16
    c4, p4 = down_block(p3, f[3])  # 16 -> 8

    bn = bottleneck(p4, f[4])

    u1 = up_block(bn, c4, f[3])  # 8 -> 16
    u2 = up_block(u1, c3, f[2])  # 16 -> 32
    u3 = up_block(u2, c2, f[1])  # 32 -> 64
    u4 = up_block(u3, c1, f[0])  # 64 -> 128

    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u4)
    model = keras.models.Model(inputs, outputs)
    return model

# Define the validation data generator
valid_path =  "stage3_train"
val_data_size = 72
valid_ids = next(os.walk(valid_path))[1][:val_data_size]  # Define valid_ids here
batch_size = 8  # Define batch_size here
aug_valid_gen = AugmentedDataGen(valid_ids, valid_path, batch_size=batch_size, image_size=128)

# Load the model weights
model_weights_path = "model/Aug_50.weights.h5"
model = UNet()
model.load_weights(model_weights_path)

# Make predictions
# index = 1  # Change this to choose a different image

index = int(input('Enter index: '))

x_val, y_val = aug_valid_gen.__getitem__(index)
predictions = model.predict(x_val)

# Visualize the results
plt.figure(figsize=(12, 6))

# Original Image
plt.subplot(1, 3, 1)
plt.imshow(x_val[0])
plt.title('Original Image')
plt.axis('off')

# Ground Truth Mask
plt.subplot(1, 3, 2)
plt.imshow(np.reshape(y_val[0] * 255, (128, 128)), cmap="gray")  # Assuming image_size is 128
plt.title('Ground Truth Mask')
plt.axis('off')

# Predicted Mask
plt.subplot(1, 3, 3)
plt.imshow(np.reshape(predictions[0] * 255, (128, 128)), cmap="gray")  # Assuming image_size is 128
plt.title('Predicted Mask')
plt.axis('off')

plt.show()
