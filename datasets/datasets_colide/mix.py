import pandas as pd
from sklearn.model_selection import train_test_split


def merge_and_split_datasets(df1, df2, output_train_path, output_test_path, random_state=None):
    """
    Объединяет два датафрейма, разделяет в соотношении 9:1 и сохраняет в CSV.

    Параметры:
    ----------
    df1 : pandas.DataFrame
        Первый датафрейм для объединения
    df2 : pandas.DataFrame
        Второй датафрейм для объединения
    output_train_path : str
        Путь для сохранения большей части данных (90%)
    output_test_path : str
        Путь для сохранения меньшей части данных (10%)
    random_state : int, optional
        Seed для воспроизводимости случайного разделения
    """
    # Объединяем датафреймы
    merged_df = pd.concat([df1, df2], ignore_index=True)

    # Разделяем на train и test в соотношении 90% к 10%
    train_df, test_df = train_test_split(
        merged_df,
        test_size=0.1,
        random_state=random_state
    )

    # Сохраняем результаты
    train_df.to_csv(output_train_path, index=False, sep="\t")
    test_df.to_csv(output_test_path, index=False, sep="\t")

    print(f"Данные успешно сохранены:\n- {output_train_path} (90% данных)\n- {output_test_path} (10% данных)")
    print(len(train_df))
    print(len(test_df))


df1 = pd.read_csv("/Users/denismazepa/Desktop/Py_projects/VKR/datasets/datatsets_from_git/train/train_ru_work.csv", sep="\t", on_bad_lines='skip')
df2 = pd.read_csv("/Users/denismazepa/Desktop/Py_projects/VKR/datasets/datatsets_from_git/test/test_ru_work.csv", sep="\t", on_bad_lines='skip')
# Вызываем функцию
merge_and_split_datasets(
    df1,
    df2,
    output_train_path='train_mixed_data.csv',
    output_test_path='test_mixed_data.csv',
    random_state=42
)