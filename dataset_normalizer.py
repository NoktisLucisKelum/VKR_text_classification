import pandas as pd


def split_and_rename_columns(dataframe: pd.DataFrame, column_names: list):
    """
    Делит каждый DataFrame в списке по символу \t и присваивает имена столбцам.

    :param dataframes: список pandas DataFrame, каждый с одной колонкой
    :param column_names: список из шести строк, представляющих имена для новых столбцов
    :return: список DataFrame, где каждая строка разделена на шесть столбцов с заданными именами
    """

    if dataframe.shape[1] != 1:
        raise ValueError("DataFrame должен содержать ровно одну колонку")

    # Разделяем столбец по символу \t
    split_df = df.iloc[:, 0].str.split('\t', expand=True)

    # Присваиваем имена столбцам
    split_df.columns = column_names
    split_df = split_df[column_names]

    return split_df


df = pd.read_csv("datasets/datatsets_from_git/train/train_ru_work.csv", on_bad_lines='skip').head(50)

column_names = ['title', 'body', 'keywords', 'correct_RGNTI1', 'RGNTI2', 'RGNTI3']

new_dataframe = split_and_rename_columns(df, column_names)
new_dataframe.to_csv("datasets/datasets_final/train_refactored.csv", index=False)


