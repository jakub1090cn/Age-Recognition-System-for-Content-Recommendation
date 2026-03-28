import os

# Base paths
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)

# Temporary files directory
TEMP_DIR = os.path.join(APP_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# Image paths
CAPTURED_IMAGE_PATH = os.path.join(TEMP_DIR, "captured_image.jpg")
FACE_DETECTED_PATH = os.path.join(TEMP_DIR, "face_detected.jpg")
PLACEHOLDER_PATH = os.path.join(APP_DIR, "placeholder.png")

# Model paths
# Defaulting to a relative path, but allowing override via environment variable
MODEL_PATH = os.environ.get(
    "MODEL_PATH", 
    os.path.join(PROJECT_ROOT, "models", "best_model.keras")
)

# Haar Cascade for face detection
CASCADE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "haarcascade_frontalface_default.xml")
# Fallback to cv2 default if local file doesn't exist
import cv2
if not os.path.exists(CASCADE_PATH):
    CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Image processing constants
IMAGE_SIZE = (224, 224)
