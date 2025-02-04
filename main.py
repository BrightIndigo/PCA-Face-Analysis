import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Ścieżka do folderu ze zdjęciami
FOLDER_PATH = "./zdj"  
IMAGE_SIZE = (100, 100)  #Skalowanie obrazów

# Funkcja do wczytywania obrazów z folderu
def load_images_from_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"Błąd: Folder {folder_path} nie istnieje!")
        return np.array([]), []

    images = []
    filenames = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Wczytanie jako skala szarości
        if img is None:
            print(f"Nie udało się wczytać pliku: {filename}")
            continue
        img = cv2.resize(img, IMAGE_SIZE)  # Skalowanie do jednolitego rozmiaru
        images.append(img.flatten())  # Spłaszczenie obrazu do wektora
        filenames.append(filename)

    if len(images) == 0:
        print("Brak obrazów w folderze!")
        return np.array([]), []

    return np.array(images), filenames

# Wczytywanie obrazów
X, filenames = load_images_from_folder(FOLDER_PATH)
if X.size == 0:
    exit()

print(f"Załadowano {len(X)} obrazów. Kształt danych: {X.shape}")

# Wykonanie PCA
n_components = 50  
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

print(f"Kształt danych po PCA: {X_pca.shape}")

# Wizualizacja
def plot_eigenfaces(pca, image_size, n_components=10):
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        if i >= n_components or i >= len(pca.components_):
            break
        eigenface = pca.components_[i].reshape(image_size)
        ax.imshow(eigenface, cmap='gray')
        ax.set_title(f"PC {i+1}")
        ax.axis('off')
    plt.show()

plot_eigenfaces(pca, IMAGE_SIZE, n_components=10)
