import os
import time

from data_generator import DataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    CSVLogger,
)

from src.config import AGE_RANGES
from src.model import create_resnet_model

from tensorflow.keras.metrics import RootMeanSquaredError
from src.prepare_data import (
    prepare_data,
)


def main():
    images_folder = r"C:\Users\user\Projekty\FaceData_2"

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_folder = f"../training_{timestamp}"

    os.makedirs(output_folder, exist_ok=True)

    models_directory = os.path.join(output_folder, "models")
    os.makedirs(models_directory, exist_ok=True)

    history_directory = os.path.join(output_folder, "history")
    os.makedirs(history_directory, exist_ok=True)

    age_ranges = AGE_RANGES

    data_directory = os.path.join(output_folder, "data")

    prepare_data(images_folder, age_ranges, directory_path=data_directory)

    # Parametry generatora
    params = {
        "dim": (224, 224),
        "batch_size": 32,
        "n_channels": 3,
        "shuffle": True,
    }

    # Stworzenie modelu z transfer learningiem
    model = create_resnet_model(input_shape=(224, 224, 3))

    # Kompilacja modelu
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="mean_squared_error",
        metrics=[RootMeanSquaredError()],
    )

    # Trenowanie modelu sekwencyjnie na przedziałach wiekowych
    for range_name in age_ranges.keys():
        print(f"\nTrenowanie na danych z przedziału: {range_name}")

        # Ścieżki do plików CSV

        train_csv = os.path.join(data_directory, f"{range_name}_train.csv")
        val_csv = os.path.join(data_directory, f"{range_name}_val.csv")

        if not os.path.exists(train_csv) or not os.path.exists(val_csv):
            print(f"Brak danych dla przedziału {range_name}")
            continue

        # Generatory danych
        training_generator = DataGenerator(train_csv, images_folder, **params)
        validation_generator = DataGenerator(val_csv, images_folder, **params)

        # Sprawdzenie, czy są dane w generatorze
        if len(training_generator) == 0:
            print(f"Brak danych treningowych dla przedziału {range_name}")
            continue

        # Callbacks
        callbacks = [
            ModelCheckpoint(
                filepath=os.path.join(models_directory, "best_model.keras"),
                monitor="val_loss",
                save_best_only=True,
            ),
            EarlyStopping(monitor="val_loss", patience=10),
            CSVLogger(
                os.path.join(history_directory, f"history_{range_name}.csv")
            ),
        ]

        # Trenowanie modelu
        epochs = 100
        model.fit(
            training_generator,
            validation_data=validation_generator,
            epochs=epochs,
            callbacks=callbacks,
        )

    test_csv = os.path.join(data_directory, "test_data.csv")
    if os.path.exists(test_csv):
        test_generator = DataGenerator(test_csv, images_folder, **params)
        test_loss, test_rmse = model.evaluate(test_generator)
        print(f"\nTest RMSE: {test_rmse}, Test MSE: {test_loss}")
    else:
        print("Brak zbioru testowego do ewaluacji.")

    model.save(os.path.join(output_folder, "final_model.keras"))


if __name__ == "__main__":
    main()
