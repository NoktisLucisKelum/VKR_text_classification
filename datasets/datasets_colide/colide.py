import pandas as pd
import numpy as np


def process_datasets(df1, df2, x_column):
    """
    Обрабатывает два датафрейма согласно заданным шагам.

    Параметры:
    df1 (pd.DataFrame): Первый датафрейм
    df2 (pd.DataFrame): Второй датафрейм (должен иметь те же столбцы, что и df1)
    x_column (str): Название столбца для фильтрации и разделения

    Возвращает:
    tuple: (обновленный df1, меньшая часть df2)
    """

    # Шаг 1: Фильтрация df2 - оставляем только строки со значениями x_column, которые есть в df1
    print(0)
    common_values = set(df1[x_column].unique()) & set(df2[x_column].unique())
    df2_filtered = df2[df2[x_column].isin(common_values)].copy()

    # Шаг 2: Разделение df2_filtered на 85% и 15% с сохранением уникальных значений
    print(1)
    unique_values = df2_filtered[x_column].unique()
    np.random.seed(42)  # для воспроизводимости
    np.random.shuffle(unique_values)

    print(2)

    split_idx = int(len(unique_values) * 0.85)
    train_values = unique_values[:split_idx]
    test_values = unique_values[split_idx:]

    print(3)

    df2_train = df2_filtered[df2_filtered[x_column].isin(train_values)]
    df2_test = df2_filtered[df2_filtered[x_column].isin(test_values)]

    print(4)

    # Шаг 3: Объединение df1 с большей частью df2 и возврат результатов
    df1_updated = pd.concat([df1, df2_train], axis=0, ignore_index=True)

    print(5)

    return df1_updated, df2_test


df1 = pd.read_csv("/Users/denismazepa/Desktop/Py_projects/VKR/datasets/datatsets_from_git/train/train_ru_work.csv", sep="\t", on_bad_lines='skip')
df2 = pd.read_csv("/Users/denismazepa/Desktop/Py_projects/VKR/datasets/datatsets_from_git/test/test_ru_work.csv", sep="\t", on_bad_lines='skip')

# Применяем функцию
df1_updated, df2_test = process_datasets(df1, df2, 'RGNTI3')

print("Обновленный df1:")
print(len(df1_updated))
print("\nМеньшая часть df2:")
print(len(df2_test))
df1_updated.to_csv("train_work_big.csv", sep="\t")
df2_test.to_csv("test_work_small.csv", sep="\t")