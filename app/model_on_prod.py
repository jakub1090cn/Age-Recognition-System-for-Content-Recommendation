import cv2
import tensorflow as tf
import numpy as np
import time
from tensorflow.keras.applications.resnet50 import preprocess_input
from taipy.gui import Gui, notify, State
from preventive_examinations import examinations_by_age

model = tf.keras.models.load_model(
    r"C:\Users\user\Projekty\Bacherols\training_20241211-100353\models\best_model.keras"
)
img_path = r"C:\Users\user\Projekty\Bacherols\app\maciek.png"

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

image_data = img_path
prediction = None
prediction_examinations = None


def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None, "Nie udało się otworzyć kamery!"

    for _ in range(10):
        ret, frame = cap.read()
        time.sleep(0.1)

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None or np.all(frame == 0):
        return None, "Nie udało się zrobić zdjęcia lub obraz jest czarny!"
    return frame, None


def save_image(image, filename="captured_image.jpg"):
    cv2.imwrite(filename, image)
    return filename


def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    if len(faces) == 0:
        return None, "Nie wykryto twarzy na zdjęciu!"

    x, y, w, h = faces[0]
    return image[y : y + h, x : x + w], None


def preprocess_image(image):
    img = cv2.resize(image, (224, 224))
    img = np.expand_dims(img, axis=0)
    return preprocess_input(img)


def predict_age(image):
    predictions = model.predict(image)
    return round(predictions[0][0])


def capture_and_predict(state):
    frame, error = capture_image()
    if error:
        notify(state, "error", error)
        print("Błąd:", error)
        return

    filename = save_image(frame)
    print(f"Zdjęcie zrobione i zapisane jako {filename}.")

    face, error = detect_face(frame)
    if error:
        notify(state, "error", error)
        print("Błąd:", error)
        return

    face_filename = save_image(face, "face_detected.jpg")
    state.image_data = face_filename
    print("Twarz wykryta i zapisana jako:", face_filename)

    try:
        preprocessed_image = preprocess_image(face)
        age = predict_age(preprocessed_image)
        group = "51+"
        match age:
            case age if 0 <= age <= 20:
                group = "0-20"
            case age if 21 <= age <= 30:
                group = "21-30"
            case age if 31 <= age <= 40:
                group = "31-40"
            case age if 41 <= age <= 50:
                group = "41-50"

        examinations_list = examinations_by_age.get(group, [])
        examinations = ", ".join(examinations_list)

        state.prediction = f"Twój wiek: {age}"
        state.prediction_examinations = f"Proponowane badania: {examinations}"

        print(f"Przewidywany wiek: {age}, Badania: {examinations}")

        notify(state, "success", "Przewidywanie zakończone!")
    except Exception as e:
        notify(state, "error", f"Błąd podczas przetwarzania: {e}")
        print("Błąd:", e)


page = """
<|text-center|
# Profilaktyka gwarantem zdrowia

<|part|render=image_data|>
<|{image_data}|image|>

<|button|label=Zrób zdjęcie|on_action=capture_and_predict|>

<|part|render=prediction|>
<|{prediction}|text|>

<|part|render=prediction_examinations|>
<|{prediction_examinations}|text|>
>
"""


def on_change(state, var_name, var_val):
    if var_name == "image_data":
        print("Nowe zdjęcie zostało zrobione!")
    elif var_name == "prediction":
        print("Nowe przewidywanie: ", var_val)
    elif var_name == "prediction_examinations":
        print("Nowe badania: ", var_val)


if __name__ == "__main__":
    Gui(page).run()
