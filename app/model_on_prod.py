from taipy.gui import Gui, notify
from preventive_examinations import examinations_by_age
from config import CAPTURED_IMAGE_PATH, FACE_DETECTED_PATH, PLACEHOLDER_PATH
from camera_service import capture_image
from image_utils import save_image, detect_face, preprocess_image
from model_service import AgePredictionModel

# Initialize the prediction model
model_service = AgePredictionModel()

# UI state variables
image_data = PLACEHOLDER_PATH
prediction = ""
prediction_examinations = ""

def capture_and_predict(state):
    """
    Action triggered by the UI to capture image, detect face, and predict age.
    """
    # 1. Capture image
    frame, error = capture_image()
    if error:
        notify(state, "error", error)
        print("Błąd kamery:", error)
        return

    # 2. Save captured image (optional, but good for debugging)
    save_image(frame, CAPTURED_IMAGE_PATH)
    print(f"Zdjęcie zrobione i zapisane w {CAPTURED_IMAGE_PATH}")

    # 3. Detect face
    face, error = detect_face(frame)
    if error:
        notify(state, "error", error)
        print("Błąd detekcji twarzy:", error)
        return

    # 4. Save detected face and update UI
    face_filename = save_image(face, FACE_DETECTED_PATH)
    state.image_data = face_filename
    print(f"Twarz wykryta i zapisana w {face_filename}")

    # 5. Predict age
    try:
        if model_service.model is None:
            notify(state, "error", "Model nie został załadowany! Sprawdź ścieżkę w config.py")
            return

        preprocessed_image = preprocess_image(face)
        age = model_service.predict(preprocessed_image)
        
        # 6. Get examinations for predicted age group
        group = "51+"
        if 0 <= age <= 20:
            group = "0-20"
        elif 21 <= age <= 30:
            group = "21-30"
        elif 31 <= age <= 40:
            group = "31-40"
        elif 41 <= age <= 50:
            group = "41-50"

        examinations_list = examinations_by_age.get(group, [])
        examinations = ", ".join(examinations_list)

        # 7. Update UI state
        state.prediction = f"Twój wiek: {age}"
        state.prediction_examinations = f"Proponowane badania: {examinations}"

        print(f"Przewidywany wiek: {age}, Badania: {examinations}")
        notify(state, "success", "Przewidywanie zakończone!")
        
    except Exception as e:
        notify(state, "error", f"Błąd podczas przetwarzania: {e}")
        print("Błąd:", e)

# Taipy GUI Page Layout
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
    """
    Log changes in state variables.
    """
    if var_name == "image_data":
        print("Nowe zdjęcie zostało wyświetlone!")
    elif var_name in ["prediction", "prediction_examinations"]:
        print(f"Aktualizacja {var_name}: {var_val}")

if __name__ == "__main__":
    Gui(page).run()
