import os
import cv2
import numpy as np
import tensorflow as tf  # Import TensorFlow to load the model
 

def predict_infection_status(model, image_path):
    # Load and preprocess the image
    img = cv2.imread(image_path)
    img = (
        cv2.resize(img, (256, 256)) / 255.0
    )  # Ensure the size matches the model's input
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make a prediction
    prediction = model.predict(img)

    # Interpret the result
    if prediction[0] > 0.5:
        print(f"{image_path}: The image is predicted to be infected.")
    else:
        print(f"{image_path}: The image is predicted to be non-infected.")



model_path = "cyst_detection_final_savedmodel.h5"  # Path to your saved model
model = tf.keras.models.load_model(model_path)  # Load the model
image_path = r"C:\Python\dataset\dataset\OTU_2D\test\image\3.JPG"

predict_infection_status(model, image_path)