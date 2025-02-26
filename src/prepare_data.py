import os
import csv
from collections import Counter

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from src.config import MAX_CLASS_COUNT


def extract_labels_and_filenames(images_folder):
    file_names = os.listdir(images_folder)

    image_extensions = [".jpg", ".jpeg", ".png"]
    file_names = [
        fn
        for fn in file_names
        if os.path.splitext(fn)[1].lower() in image_extensions
    ]

    labels = []
    valid_file_names = []
    for fn in file_names:
        label_str = fn.split("_")[0]
        try:
            label = int(label_str)
            labels.append(label)
            valid_file_names.append(fn)
        except ValueError:
            print(f"Nie można wyodrębnić etykiety z pliku: {fn}")

    return valid_file_names, labels


def divide_data_by_age_range(file_names, labels, age_ranges):
    data_by_age_range = {}
    for range_name, (age_min, age_max) in age_ranges.items():
        indices = [
            i for i, age in enumerate(labels) if age_min <= age <= age_max
        ]
        range_file_names = [file_names[i] for i in indices]
        range_labels = [labels[i] for i in indices]
        data_by_age_range[range_name] = (range_file_names, range_labels)
    return data_by_age_range


def save_data_to_csv(file_names, labels, csv_path):
    directory = os.path.dirname(csv_path)
    os.makedirs(directory, exist_ok=True)

    with open(csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["file_name", "label"])
        for fn, label in zip(file_names, labels):
            writer.writerow([fn, label])


def prepare_data(images_folder, age_ranges, directory_path):
    file_names, labels = extract_labels_and_filenames(images_folder)
    balance_data_with_augmentation(
        file_names, labels, images_folder, max_class_count=MAX_CLASS_COUNT
    )

    (
        file_names_train_val,
        file_names_test,
        labels_train_val,
        labels_test,
    ) = train_test_split(file_names, labels, test_size=0.15, random_state=42)

    save_data_to_csv(
        file_names_test,
        labels_test,
        os.path.join(directory_path, "test_data.csv"),
    )

    (
        file_names_train,
        file_names_val,
        labels_train,
        labels_val,
    ) = train_test_split(
        file_names_train_val,
        labels_train_val,
        test_size=0.1765,
        random_state=42,
    )

    data_by_age_range_train = divide_data_by_age_range(
        file_names_train, labels_train, age_ranges
    )
    data_by_age_range_val = divide_data_by_age_range(
        file_names_val, labels_val, age_ranges
    )

    for range_name in age_ranges.keys():
        fn_train_range, labels_train_range = data_by_age_range_train.get(
            range_name, ([], [])
        )
        save_data_to_csv(
            fn_train_range,
            labels_train_range,
            os.path.join(directory_path, f"{range_name}_train.csv"),
        )

        fn_val_range, labels_val_range = data_by_age_range_val.get(
            range_name, ([], [])
        )
        save_data_to_csv(
            fn_val_range,
            labels_val_range,
            os.path.join(directory_path, f"{range_name}_val.csv"),
        )


def augment_image(image):
    """
    Tworzy jedną zaugmentowaną wersję obrazu twarzy przy użyciu OpenCV.

    Args:
        image (numpy.ndarray): Obraz wejściowy w formie macierzy NumPy.

    Returns:
        numpy.ndarray: Zaugmentowany obraz.
    """
    augmented = image.copy()
    height, width = image.shape[:2]

    # Ograniczony obrót
    angle = np.random.uniform(-10, 10)
    matrix = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1.0)
    augmented = cv2.warpAffine(augmented, matrix, (width, height))

    # Drobne zmiany jasności
    if np.random.rand() > 0.5:
        alpha = np.random.uniform(0.8, 1.2)  # Współczynnik jasności
        augmented = cv2.convertScaleAbs(augmented, alpha=alpha, beta=0)

    # Odbicie lustrzane
    if np.random.rand() > 0.5:
        augmented = cv2.flip(augmented, 1)

    # Drobne przesunięcie
    max_shift = int(0.05 * min(width, height))
    dx = np.random.randint(-max_shift, max_shift)
    dy = np.random.randint(-max_shift, max_shift)
    translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    augmented = cv2.warpAffine(augmented, translation_matrix, (width, height))

    # Losowy szum Gaussowski
    if np.random.rand() > 0.5:
        row, col, ch = augmented.shape
        mean = 0
        sigma = np.random.uniform(
            10, 25
        )  # Można dostosować zakres dla różnych intensywności szumu
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        noisy = np.uint8(np.clip(augmented + gauss, 0, 255))
        augmented = noisy

    return augmented


def balance_data_with_augmentation(
    file_names, labels, images_folder, max_class_count=None
):
    """
    Balansuje dane zdjęć twarzy poprzez augmentację. Automatycznie generuje tyle augmentacji, ile jest potrzebne.

    Args:
        file_names (list): Lista nazw plików.
        labels (list): Lista etykiet odpowiadających plikom.
        images_folder (str): Folder z oryginalnymi obrazami.
        max_class_count (int): Maksymalna liczba obrazów w klasie po augmentacji.

    Returns:
        (list, list): Zbalansowane nazwy plików i etykiety.
    """
    target_folder = images_folder
    os.makedirs(target_folder, exist_ok=True)

    # Policz liczebność każdej klasy
    class_counts = Counter(labels)
    max_count = max(class_counts.values())

    # Nowe listy na zbalansowane dane
    balanced_file_names = []
    balanced_labels = []

    for label in class_counts:
        class_file_names = [
            fn for fn, lbl in zip(file_names, labels) if lbl == label
        ]
        balanced_file_names.extend(class_file_names)
        balanced_labels.extend([label] * len(class_file_names))

        # Dodawanie augmentacji dla niedoreprezentowanych klas
        if class_counts[label] < max_count:
            augment_count = max_count - class_counts[label]

            while augment_count > 0:
                for file_name in class_file_names:
                    if augment_count <= 0:
                        break

                    image_path = os.path.join(images_folder, file_name)
                    image = cv2.imread(image_path)

                    # Wygenerowanie jednej augmentacji
                    aug_image = augment_image(image)

                    # Zapisanie zaugmentowanego obrazu
                    augmented_name = (
                        f"{label}_augmented_{len(balanced_file_names)}.jpg"
                    )
                    augmented_path = os.path.join(
                        target_folder, augmented_name
                    )
                    cv2.imwrite(augmented_path, aug_image)

                    # Dodanie danych augmentacji do listy
                    balanced_file_names.append(augmented_name)
                    balanced_labels.append(label)
                    augment_count -= 1

        elif class_counts[label] > max_class_count:
            class_file_names = class_file_names[:max_class_count]
            balanced_file_names.extend(class_file_names)
            balanced_labels.extend([label] * len(class_file_names))

    return balanced_file_names, balanced_labels
