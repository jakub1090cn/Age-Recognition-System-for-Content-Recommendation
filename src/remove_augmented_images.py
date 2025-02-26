import os


def remove_augmented_images(folder_path):
    """
    Usuwa zdjęcia z folderu, które w nazwie mają słowo 'augmented'.

    Args:
        folder_path (str): Ścieżka do folderu z obrazami.
    """
    # Sprawdzamy wszystkie pliki w folderze
    for filename in os.listdir(folder_path):
        # Sprawdzamy, czy w nazwie pliku jest słowo 'augmented'
        if (
            "augmented" in filename.lower()
        ):  # Używamy .lower(), żeby porównanie było niewrażliwe na wielkość liter
            file_path = os.path.join(folder_path, filename)
            try:
                # Usuwamy plik
                os.remove(file_path)
                print(f"Usunięto: {filename}")
            except Exception as e:
                print(f"Nie udało się usunąć pliku {filename}: {e}")


folder_path = r"C:\Users\user\Projekty\FaceData_2"
remove_augmented_images(folder_path)
