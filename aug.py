# Imports
import os
import random

import numpy as np
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore # Importation ajoutée

# Seeding
seed = 2019
random.seed = seed
np.random.seed = seed
tf.random.set_seed(seed)  # Mise à jour de la syntaxe pour la version TensorFlow 2.x


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
            _mask_image = np.expand_dims(_mask_image, axis=-1)
            mask = np.maximum(mask, _mask_image)

        # Normalisation
        image = image / 255.0
        mask = mask / 255.0

        return image, mask

    def __getitem__(self, index):
        if (index+1)*self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index*self.batch_size

        files_batch = self.ids[index*self.batch_size: (index+1)*self.batch_size]

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


# Model training variables
train_path = "/stage3_train"
image_size = 128
epochs = 50
batch_size = 8

# Training Ids
train_ids = next(os.walk(train_path))[1]

# Validation Data Size
val_data_size = 72

valid_ids = train_ids[:val_data_size]
train_ids = train_ids[val_data_size:]

# Data augmentation parameters
data_gen_args = dict(
    rotation_range=90,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)


# Augmented data generation for training and validation
train_image_datagen = ImageDataGenerator(**data_gen_args)
valid_image_datagen = ImageDataGenerator()

aug_train_gen = AugmentedDataGen(train_ids, train_path, batch_size=batch_size, image_size=image_size, image_datagen=train_image_datagen)
aug_valid_gen = AugmentedDataGen(valid_ids, train_path, batch_size=batch_size, image_size=image_size, image_datagen=valid_image_datagen)


gen = AugmentedDataGen(train_ids, train_path, batch_size=batch_size, image_size=image_size)
x, y = gen.__getitem__(0)
print(x.shape, y.shape)

if len(x) == 0:
    print("Error: Empty batch for training data.")
else:
    r = random.randint(0, len(x)-1)
    print("Random index:", r)
    # Reste du code...


r = random.randint(0, len(x)-1)


def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
    return c, p


def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    us = keras.layers.UpSampling2D((2, 2))(x)
    concat = keras.layers.Concatenate()([us, skip])
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c


def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

# Modèle UNet
def UNet():
    f = [16, 32, 64, 128, 256]
    inputs = keras.layers.Input((image_size, image_size, 3))

    p0 = inputs
    c1, p1 = down_block(p0, f[0])  # 128 -> 64
    c2, p2 = down_block(p1, f[1])  # 64 -> 32
    c3, p3 = down_block(p2, f[2])  # 32 -> 16
    c4, p4 = down_block(p3, f[3])  # 16->8

    bn = bottleneck(p4, f[4])

    u1 = up_block(bn, c4, f[3])  # 8 -> 16
    u2 = up_block(u1, c3, f[2])  # 16 -> 32
    u3 = up_block(u2, c2, f[1])  # 32 -> 64
    u4 = up_block(u3, c1, f[0])  # 64 -> 128

    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u4)
    model = keras.models.Model(inputs, outputs)
    return model


model = UNet()

from tensorflow.keras.metrics import MeanIoU # type: ignore

iou_metric = MeanIoU(num_classes=2)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc", iou_metric])
model.summary()

aug_train_gen = AugmentedDataGen(train_ids, train_path, image_size=image_size, batch_size=batch_size)
aug_valid_gen = AugmentedDataGen(valid_ids, train_path, image_size=image_size, batch_size=batch_size)

total_epochs = epochs

train_loss = []
train_acc = []
train_iou = []
val_loss = []
val_acc = []
val_iou = []

# Model Training over specified number of epochs
for epoch in range(total_epochs):
    # Calculate steps per epoch for the current epoch
    train_steps = len(train_ids) // batch_size
    valid_steps = len(valid_ids) // batch_size

    # Train the model (one epoch)
    history = model.fit(aug_train_gen, validation_data=aug_valid_gen, steps_per_epoch=train_steps, validation_steps=valid_steps, epochs=1)
    train_loss.append(history.history['loss'][0])
    train_acc.append(history.history['acc'][0])
    train_iou.append(history.history['mean_io_u'][0])
    val_loss.append(history.history['val_loss'][0])
    val_acc.append(history.history['val_acc'][0])
    val_iou.append(history.history['val_mean_io_u'][0])

    # Print the metrics for the current epoch
    print(f"Epoch {epoch + 1}/{total_epochs} - Loss: {train_loss[epoch]:.4f} - Acc: {train_acc[epoch]:.4f} - IoU: {train_iou[epoch]:.4f} - Val Loss: {val_loss[epoch]:.4f} - Val Acc: {val_acc[epoch]:.4f} - Val IoU: {val_iou[epoch]:.4f}")

# Change directories here to yours, for saving model data
# Sauvegarde des poids du modèle
model.save_weights(f"model/Aug_{epochs}.weights.h5")
model.save(f"model/aug{epochs}.h5")
model.save('trained_model.keras')

# Visualisation des résultats
# train_loss = history.history['loss']
# train_acc = history.history['acc']
# train_iou = history.history['mean_io_u']
# val_loss = history.history['val_loss']
# val_acc = history.history['val_acc']
# val_iou = history.history['val_mean_io_u']

# Tracé
plt.figure()
plt.plot(train_loss)
plt.plot(val_loss)
plt.legend(['Training loss', 'Validation loss'])
plt.title('Loss')
plt.show()

plt.figure()
plt.plot(train_acc)
plt.plot(val_acc)
plt.legend(['Training accuracy', 'Validation accuracy'])
plt.title('Accuracy')
plt.show()

plt.figure()
plt.plot(train_iou)
plt.plot(val_iou)
plt.legend(['Training IoU', 'Validation IoU'])
plt.title('IoU')
plt.show()

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{total_epochs} - Loss: {train_loss[-1]:.4f} - Acc: {train_acc[-1]:.4f} - IoU: {train_iou[-1]:.4f} - Val Loss: {val_loss[-1]:.4f} - Val Acc: {val_acc[-1]:.4f} - Val IoU: {val_iou[-1]:.4f}")


# FIX Deep learning model prediction
# Dataset for prediction
x, y = aug_valid_gen.__getitem__(1)
result = model.predict(x)
result = result > 0.5

all_predictions = []

# FIX
# Boucle à travers le générateur de validation pour obtenir les prédictions pour chaque lot
for i in range(len(aug_valid_gen)):
    x_val, y_val = aug_valid_gen.__getitem__(i)
    predictions = model.predict(x_val)
    all_predictions.extend(predictions)

    # Code de tracé
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Tracé du masque de vérité terrain
    axes[0].imshow(np.reshape(y_val[0] * 255, (image_size, image_size)), cmap="gray")
    axes[0].set_title(f'Ground Truth Mask {i + 1}')

    # Tracé du masque prédit
    axes[1].imshow(np.reshape(predictions[0] * 255, (image_size, image_size)), cmap="gray")
    axes[1].set_title(f'Predicted Mask {i + 1}')

    plt.show()
