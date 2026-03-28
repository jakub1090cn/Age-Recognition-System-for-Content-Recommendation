import tensorflow as tf
import os
from .config import MODEL_PATH

class AgePredictionModel:
    """
    Service for loading the trained model and predicting age.
    """
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.model = None
        
        if os.path.exists(self.model_path):
            try:
                self.model = tf.keras.models.load_model(self.model_path)
                print(f"Model loaded successfully from {self.model_path}")
            except Exception as e:
                print(f"Error loading model from {self.model_path}: {e}")
        else:
            print(f"Warning: Model file not found at {self.model_path}")

    def predict(self, preprocessed_image):
        """
        Predicts age from a preprocessed image.
        """
        if self.model is None:
            raise Exception("Model not loaded. Please ensure the model file exists.")
        
        predictions = self.model.predict(preprocessed_image)
        return round(predictions[0][0])
