import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from .config import CASCADE_PATH, IMAGE_SIZE

# Load the face cascade once
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

def save_image(image, filename):
    """
    Saves an image to the specified filename.
    """
    cv2.imwrite(filename, image)
    return filename

def detect_face(image):
    """
    Detects a face in the image and returns the cropped face.
    Returns (face_image, error_message)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    
    if len(faces) == 0:
        return None, "Nie wykryto twarzy na zdjęciu!"

    # Take the first face detected
    x, y, w, h = faces[0]
    return image[y : y + h, x : x + w], None

def preprocess_image(image):
    """
    Resizes and preprocesses the image for the ResNet model.
    """
    img = cv2.resize(image, IMAGE_SIZE)
    img = np.expand_dims(img, axis=0)
    return preprocess_input(img)
