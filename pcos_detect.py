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


def test_folder(model, folder_path):
    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        if (
            filename.endswith(".JPG")
            or filename.endswith(".jpeg")
            or filename.endswith(".png")
        ):
            image_path = os.path.join(folder_path, filename)
            predict_infection_status(model, image_path)


# Example usage
model_path = "cyst_detection_final_savedmodel.h5"  # Path to your saved model
model = tf.keras.models.load_model(model_path)  # Load the model
test_folder_path = r"C:\Python\dataset\dataset\OTU_2D\test\image"
test_folder(model, test_folder_path)