import os
import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications.resnet50 import preprocess_input


class DataGenerator(Sequence):
    def __init__(
        self,
        csv_file,
        images_folder,
        batch_size=32,
        dim=(224, 224),
        n_channels=3,
        shuffle=True,
    ):
        self.csv_file = csv_file
        self.images_folder = images_folder
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle

        self.file_names, self.labels = self._load_data_from_csv()
        self.indexes = np.arange(len(self.file_names))
        self.on_epoch_end()

    def _load_data_from_csv(self):
        import csv

        file_names = []
        labels = []
        with open(self.csv_file, mode="r", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                file_names.append(row["file_name"])
                labels.append(float(row["label"]))
        return file_names, labels

    def __len__(self):
        return int(np.ceil(len(self.file_names) / self.batch_size))

    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = min((index + 1) * self.batch_size, len(self.file_names))
        indexes = self.indexes[start_index:end_index]

        batch_file_names = [self.file_names[k] for k in indexes]
        batch_labels = [self.labels[k] for k in indexes]

        X, y = self.__data_generation(batch_file_names, batch_labels)

        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_file_names, batch_labels):
        X = np.empty(
            (len(batch_file_names), *self.dim, self.n_channels),
            dtype=np.float32,
        )
        y = np.empty((len(batch_labels),), dtype=np.float32)

        for i, (file_name, label) in enumerate(
            zip(batch_file_names, batch_labels)
        ):
            img_path = os.path.join(self.images_folder, file_name)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Nie można wczytać obrazu: {img_path}")
                continue

            img = cv2.resize(img, self.dim)

            if len(img.shape) == 2 or img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            img = img.astype(np.float32)
            img = preprocess_input(img)

            X[i] = img
            y[i] = label

        return X, y
