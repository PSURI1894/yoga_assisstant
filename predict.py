import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("yoga_pose_model.h5")

# Pose labels
pose_labels = ['Downdog', 'Goddess','Plank',  'Tree', 'Warrior2']

def preprocess_image(image_path):
    """ Load, resize, and normalize the image """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = cv2.resize(image, (128, 128))  # Resize to match model input
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict_pose(image_path):
    """ Predict yoga pose from image """
    processed_img = preprocess_image(image_path)
    prediction = model.predict(processed_img)  # Get model predictions
    predicted_pose = pose_labels[np.argmax(prediction)]  # Get highest confidence class
    return predicted_pose
