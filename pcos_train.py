import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


def load_images_and_masks(image_dir, mask_dir):
    images, masks = [], []
    for filename in os.listdir(image_dir):
        if filename.endswith(".JPG"):
            img_path = os.path.join(image_dir, filename)
            mask_path = os.path.join(mask_dir, filename.replace(".JPG", ".png"))

            img = cv2.imread(img_path)
            img = cv2.resize(img, (256, 256)) / 255.0

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (256, 256)) / 255.0

            images.append(img)
            masks.append(mask)

    return np.array(images), np.array(masks)


# Load the dataset
train_image_dir = r"C:\Python\dataset\dataset\OTU_2D\train\train_image"
train_label_dir = r"C:\Python\dataset\dataset\OTU_2D\train\train_label\label"

images, masks = load_images_and_masks(train_image_dir, train_label_dir)


# Convert masks to binary labels
labels = np.array([1 if np.max(mask) > 0 else 0 for mask in masks])


def create_synthetic_non_infected(images, num_augmented_images=1):
    # Initialize the ImageDataGenerator with various transformations
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    synthetic_images = []
    for image in images:
        # Reshape image to add batch dimension
        image = np.expand_dims(image, 0)

        # Generate augmented images
        for _ in range(num_augmented_images):
            augmented_iter = datagen.flow(image, batch_size=1)
            augmented_image = next(augmented_iter)[0]
            synthetic_images.append(augmented_image)

    return np.array(synthetic_images)


# Create synthetic non-infected images
synthetic_non_infected_images = create_synthetic_non_infected(
    images, num_augmented_images=2
)

# Combine infected and synthetic non-infected images
combined_images = np.concatenate((images, synthetic_non_infected_images), axis=0)
combined_labels = np.concatenate(
    (np.ones(len(images)), np.zeros(len(synthetic_non_infected_images))), axis=0
)
X_train, X_val, y_train, y_val = train_test_split(
    combined_images, combined_labels, test_size=0.2, random_state=42
)


def create_model(input_shape=(256, 256, 3)):
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
    return model


# Create and summarize the model
model = create_model()
model.summary()

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

# Train the model
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=10,
)

# Save the model in HDF5 format

# Save the model in TensorFlow SavedModel format
tf.keras.models.save_model(model, "cyst_detection_final_savedmodel.h5")

# Evaluate on validation set
val_loss, val_accuracy = model.evaluate(X_val, y_val)

print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc="upper left")

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc="upper left")

plt.tight_layout()
plt.show()
