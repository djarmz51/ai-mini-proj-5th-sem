import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split



from tensorflow.keras.callbacks import EarlyStopping
#for stopping extra epoch whenever accuracy crosses 100%
early_stopping = EarlyStopping(
    monitor='val_loss',  # Watch validation loss
    patience=5,          # Stop after 5 epochs without improvement
    restore_best_weights=True
)
# image dimensions....224 px works well with cnn
img_height = 224
img_width = 224
batch_size = 32

# Directory paths
train_image_dir = 'C:\\Users\\djarm_sb\\Downloads\\dataset\\dataset\\OTU_2D\\train\\train_image'
train_label_dir = 'C:\\Users\\djarm_sb\\Downloads\\dataset\\dataset\\OTU_2D\\train\\train_label\\label'

test_image_dir = 'C:\\Users\\djarm_sb\\Downloads\\dataset\\dataset\\OTU_2D\\test\\image'
test_label_dir = 'C:\\Users\\djarm_sb\\Downloads\\dataset\\dataset\\OTU_2D\\test\\label\\black_white'

# Function to load images and labels
def load_data(image_dir, label_dir):
    image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]
    label_paths = [os.path.join(label_dir, fname) for fname in os.listdir(label_dir)]
    
    images = []
    labels = []
    
    for img_path, label_path in zip(image_paths, label_paths):
        # Load image and label (mask)
        image = load_img(img_path, target_size=(img_height, img_width))
        label = load_img(label_path, target_size=(img_height, img_width), color_mode='grayscale')
        
        image = img_to_array(image) / 255.0  # Normalizing to [0, 1]
        label = img_to_array(label) / 255.0  # Normalizing [0, 1], black=0, white=1
        
        # Check if there's a cyst in the label
        cyst_present = np.any(label > 0.5)  # If cyst present
        
        images.append(image)
        labels.append(cyst_present)
    
    return np.array(images), np.array(labels)

# Load the train and test data
train_images, train_labels = load_data(train_image_dir, train_label_dir)
test_images, test_labels = load_data(test_image_dir, test_label_dir)
# Split the training data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42#random any number,used during to spliting variation
)

# Data augmentation for training data using ImageDataGenerator
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Create data generators
train_generator = train_datagen.flow(
    train_images, train_labels,
    batch_size=batch_size
)

val_datagen = ImageDataGenerator()  # No augmentation for validation data
val_generator = val_datagen.flow(
    val_images, val_labels,
    batch_size=batch_size
)

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Dropout to prevent overfitting
    layers.Dense(1, activation='sigmoid')  # Sigmoid for binary classification (cyst/no cyst)
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model with validation data
model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    callbacks=[early_stopping]
)
#current name of model coded while vc 
# Save the trained model
model.save('discords_a.h5')

# Evaluate the model on the test set...again separated from training/validation
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

# Predict on the test set and calculate custom accuracy
predictions = model.predict(test_images)
predictions = (predictions > 0.5).astype(int)  # Convert probabilities to binary labels

# Compare predictions with true labels
correct_count = np.sum(predictions.flatten() == test_labels)
custom_accuracy = correct_count / len(test_labels) * 100
print(f'Model Custom Test Accuracy: {custom_accuracy:.2f}%')
