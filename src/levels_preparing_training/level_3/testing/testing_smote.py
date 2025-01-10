from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import numpy as np

# Создаем искусственный несбалансированный набор данных
X, y = make_classification(n_classes=2, class_sep=2,
                           weights=[0.9, 0.1], # 90% одного класса, 10% другого
                           n_informative=2, n_redundant=0, # Исправлено: сумма меньше n_features
                           flip_y=0, n_features=2,
                           n_clusters_per_class=1,
                           n_samples=200, random_state=42)

# Визуализируем данные до применения SMOTE
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label='Класс 0', alpha=0.5)
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='Класс 1', alpha=0.5)
plt.title("До применения SMOTE")
plt.legend()
plt.show()

# Применяем SMOTE для создания синтетических данных
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Визуализируем данные после применения SMOTE
plt.scatter(X_resampled[y_resampled == 0][:, 0], X_resampled[y_resampled == 0][:, 1], label='Класс 0', alpha=0.5)
plt.scatter(X_resampled[y_resampled == 1][:, 0], X_resampled[y_resampled == 1][:, 1], label='Класс 1', alpha=0.5)
plt.title("После применения SMOTE")
plt.legend()
plt.show()

# Выводим количество примеров до и после применения SMOTE
print("Количество примеров до SMOTE:")
print(f"Класс 0: {np.sum(y == 0)}")
print(f"Класс 1: {np.sum(y == 1)}")

print("\nКоличество примеров после SMOTE:")
print(f"Класс 0: {np.sum(y_resampled == 0)}")
print(f"Класс 1: {np.sum(y_resampled == 1)}")
