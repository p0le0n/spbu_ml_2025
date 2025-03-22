import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

dataset_path = "./dataset"

def load_images_from_folder(folder, label, image_size=(64, 64), sample_size=200):
    """
    Загрузка изображений из папки с выборкой sample_size изображений.
    Приведение изображений к заданному размеру и маркировка.
    """
    images = []
    labels = []
    files = os.listdir(folder)
    # Выбираем первые sample_size изображений
    files = files[:sample_size] if len(files) > sample_size else files
    for filename in files:
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            # Изменяем размер изображения
            img = cv2.resize(img, image_size)
            # Преобразуем изображение из BGR (OpenCV) в RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append(label)
    return images, labels

# Задаем пути к папкам с кошками и собаками
cats_folder = os.path.join(dataset_path, "cats")
dogs_folder = os.path.join(dataset_path, "dogs")

# Загружаем изображения (здесь выбираем по 300 изображений на класс для иллюстративности)
cat_images, cat_labels = load_images_from_folder(cats_folder, label=0, image_size=(64, 64), sample_size=300)
dog_images, dog_labels = load_images_from_folder(dogs_folder, label=1, image_size=(64, 64), sample_size=300)

# Объединяем данные и метки
images = np.array(cat_images + dog_images)
labels = np.array(cat_labels + dog_labels)

# Преобразуем изображения в векторное представление (flatten)
n_samples = images.shape[0]
data = images.reshape(n_samples, -1)  # shape: (n_samples, 64*64*3)

# Разбиваем данные на обучающую и тестовую выборки (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42, stratify=labels
)

# Стандартизуем данные (для SVM и PCA)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Обучаем классификатор (SVM с линейным ядром) на исходных изображениях
svc = SVC(kernel='linear', random_state=42)
svc.fit(X_train_scaled, y_train)
y_pred = svc.predict(X_test_scaled)

acc_raw = accuracy_score(y_test, y_pred)
print("Точность классификации на исходных изображениях:", acc_raw)
print("\nОтчет по классификации:\n", classification_report(y_test, y_pred))

# 3. Применяем PCA и обучаем классификатор для разных чисел компонент.
components_list = list(range(10, 500, 50))
accuracy_list = []

for n_comp in components_list:
    pca = PCA(n_components=n_comp, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    svc_pca = SVC(kernel='linear', random_state=42)
    svc_pca.fit(X_train_pca, y_train)
    y_pred_pca = svc_pca.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred_pca)
    accuracy_list.append(acc)
    print(f"PCA компонентов: {n_comp}, точность: {acc}")

# 4. Строим график зависимости точности от числа PCA-компонент
plt.figure(figsize=(10, 6))
plt.plot(components_list, accuracy_list, marker='o')
plt.xlabel("Число PCA-компонент")
plt.ylabel("Точность классификации")
plt.title("Зависимость точности от числа PCA-компонент")
plt.show()

# 5. Определяем, сколько компонент нужно для объяснения 90% дисперсии
pca_full = PCA(random_state=42)
pca_full.fit(X_train_scaled)
cum_variance = np.cumsum(pca_full.explained_variance_ratio_)
n_components_90 = np.argmax(cum_variance >= 0.90) + 1
print("Для объяснения 90% дисперсии требуется", n_components_90, "компонент.")

# 6. Строим график зависимости процента объясненной дисперсии от числа компонент
plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, len(cum_variance)+1), cum_variance, marker='.')
plt.xlabel("Число компонент")
plt.ylabel("Накопленная объясненная дисперсия")
plt.title("Накопленная объясненная дисперсия от числа компонент")
plt.axhline(y=0.90, color='r', linestyle='--', label="90% дисперсии")
plt.legend()
plt.show()

# 7. Отрисовываем первые 10 главных компонент
# Приводим компоненты обратно к размерности изображения (64x64x3)
n_components_to_plot = min(10, pca_full.components_.shape[0])
plt.figure(figsize=(15, 6))
for i in range(n_components_to_plot):
    plt.subplot(2, 5, i+1)
    comp = pca_full.components_[i]
    comp_image = comp.reshape(64, 64, 3)
    # Нормализация для корректного отображения
    comp_image = (comp_image - comp_image.min()) / (comp_image.max() - comp_image.min())
    plt.imshow(comp_image)
    plt.title(f"PC {i+1}")
    plt.axis('off')
plt.suptitle("Первые 10 главных компонент")
plt.show()
