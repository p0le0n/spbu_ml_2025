# Импорт необходимых библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Настройка стиля графиков
sns.set_theme(style='whitegrid')

# === 1. Загрузка и первичный анализ данных ===
data = pd.read_csv('Building_Permits.csv')

# Вывод общей информации о датасете
print("Размер датасета:", data.shape)
print("Список столбцов:\n", data.columns.tolist())
print("Первые 5 строк:\n", data.head())
print("Информация о типах данных:")
data.info()
print("Статистическое описание числовых признаков:")
print(data.describe())

# Проверка пропущенных значений (отсортировано по количеству пропусков)
missing = data.isnull().sum().sort_values(ascending=False)
print("Пропущенные значения:\n", missing[missing > 0])

# Визуализация распределения по столбцу "Current Status"
plt.figure(figsize=(10,6))
sns.countplot(y='Current Status', data=data, 
              order=data['Current Status'].value_counts().index)
plt.title('Распределение статусов разрешений')
plt.show()

# Гистограмма распределения "Estimated Cost"
plt.figure(figsize=(10,6))
cost_log = np.log1p(data['Estimated Cost'].dropna())
sns.histplot(cost_log, bins=50, kde=True)
plt.title('Log-распределение оценочной стоимости (Estimated Cost)')
plt.xlabel('log(Estimated Cost + 1)')
plt.show()

# Гистограмма для "Number of Proposed Stories"
plt.figure(figsize=(10,6))
sns.histplot(data['Number of Proposed Stories'].dropna(), bins=20, kde=False)
plt.title('Распределение количества предлагаемых этажей')
plt.xlabel('Number of Proposed Stories')
plt.show()

# === 4. Очистка и обработка данных ===

# Преобразование столбцов с датами в тип datetime
date_columns = ['Permit Creation Date', 'Current Status Date', 'Filed Date', 
                'Issued Date', 'Completed Date', 'First Construction Document Date', 
                'Permit Expiration Date']
for col in date_columns:
    data[col] = pd.to_datetime(data[col], errors='coerce')

# Удаляем столбцы, которые не несут полезной информации для модели
cols_to_drop = ['Permit Number', 'Record ID', 'Location', 'Street Name', 
                'Street Name Suffix', 'Street Number', 'Street Number Suffix',
                'Block', 'Lot', 'Unit', 'Unit suffix', 'Description', 
                'Permit Type Definition', 'Existing Construction Type Description', 
                'Proposed Construction Type Description']
data.drop(columns=cols_to_drop, inplace=True, errors='ignore')

# Создадим бинарную целевую переменную: 1, если Current Status == "complete", иначе 0
data['is_completed'] = data['Current Status'].str.lower().str.strip() \
                               .apply(lambda x: 1 if x == 'complete' else 0)

# Выберем признаки для построения модели
# Для примера возьмём: 'Permit Type', 'Estimated Cost', 'Number of Proposed Stories',
# 'Supervisor District' и 'Zipcode'
features = ['Permit Type', 'Estimated Cost', 'Number of Proposed Stories', 
            'Supervisor District', 'Zipcode']
target = 'is_completed'

# Проверяем наличие пропусков в выбранных признаках
print("Пропуски в выбранных признаках:\n", data[features].isnull().sum())

# Для числовых признаков заполним пропуски медианой, для категориальных — модой
num_features = ['Estimated Cost', 'Number of Proposed Stories']
cat_features = ['Permit Type', 'Supervisor District', 'Zipcode']

for col in num_features:
    data[col].fillna(data[col].median(), inplace=True)

for col in cat_features:
    data[col].fillna(data[col].mode()[0], inplace=True)
    data[col] = data[col].astype(str)

# Применим one-hot encoding для категориальных признаков
data_encoded = pd.get_dummies(data[features], drop_first=True)

# Соберём итоговую матрицу признаков
X = pd.concat([data_encoded, data[num_features]], axis=1)
y = data[target]

print("Размер матрицы признаков:", X.shape)
print("Распределение целевой переменной:\n", y.value_counts())

# === 5. Построение предсказательной модели и 6. Оценка качества ===

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Разбиваем данные на обучающую и тестовую выборки (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Инициализируем модель RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Обучение модели
rf_model.fit(X_train, y_train)

# Предсказание на тестовой выборке
y_pred = rf_model.predict(X_test)

# Вычисляем метрику точности и выводим отчёт по классификации
acc = accuracy_score(y_test, y_pred)
print("Точность модели на тестовой выборке:", acc)
print("\nОтчет по классификации:\n", classification_report(y_test, y_pred))

# Строим матрицу ошибок
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Предсказано')
plt.ylabel('Истинное значение')
plt.title('Матрица ошибок')
plt.show()


