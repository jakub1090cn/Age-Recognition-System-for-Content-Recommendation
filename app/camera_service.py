import cv2
import time
import numpy as np

def capture_image():
    """
    Captures a frame from the default camera.
    Returns (frame, error_message)
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None, "Nie udało się otworzyć kamery!"

    # Warm up the camera (read some frames and wait a bit)
    for _ in range(10):
        ret, frame = cap.read()
        time.sleep(0.1)

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None or np.all(frame == 0):
        return None, "Nie udało się zrobić zdjęcia lub obraz jest czarny!"
    
    return frame, None
