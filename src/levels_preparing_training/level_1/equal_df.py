import pandas as pd


def select_100_per_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Возвращает новый DataFrame, где для каждого уникального значения в столбце group_col
    выбирается ровно 100 строк (с повторениями, если в группе < 100 строк).

    Параметры:
        df (pd.DataFrame): исходный DataFrame.
        group_col (str): имя столбца, по которому происходит группировка.

    Возвращаемое значение:
        pd.DataFrame: результирующий DataFrame из сэмплированных 100 строк на группу.
    """
    unique_vals = df[group_col].unique()
    sampled_frames = []

    for val in unique_vals:
        group_df = df[df[group_col] == val]

        # Если строк ≥ 100, выбираем ровно 100
        if len(group_df) >= 350:
            sampled_part = group_df.sample(n=350, random_state=42)
        # иначе если строк меньше 100, берём все
        else:
            sampled_part = group_df

        sampled_frames.append(sampled_part)

    # Объединяем результаты в один DataFrame
    result = pd.concat(sampled_frames, ignore_index=True)
    return result


# def select_100_per_group(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
#     """
#     Возвращает новый DataFrame, где для каждого уникального значения в столбце group_col
#     выбирается ровно 100 строк (с повторениями, если в группе < 100 строк).
#
#     Параметры:
#         df (pd.DataFrame): исходный DataFrame.
#         group_col (str): имя столбца, по которому происходит группировка.
#
#     Возвращаемое значение:
#         pd.DataFrame: результирующий DataFrame из сэмплированных 100 строк на группу.
#     """
#     unique_values = df[col_name].unique()
#
#     # Создаём пустой DataFrame с теми же колонками
#     new_df = pd.DataFrame(columns=df.columns)
#
#     for val in unique_values:
#         # Выбираем все строки, где col_name == val
#         subset = df[df[col_name] == val]
#
#         if len(subset) < 600:
#             # Если встречается < 600 раз, берём все строки
#             new_df = pd.concat([new_df, subset], ignore_index=True)
#         else:
#             # Иначе берём 10% от общего количества
#             sample_size = int(0.1 * len(subset))
#             sample_df = subset.sample(n=sample_size, random_state=42)
#             new_df = pd.concat([new_df, sample_df], ignore_index=True)
#
#     return new_df

# df_level_1 = pd.read_csv(
#     "/Users/denismazepa/Desktop/Py_projects/VKR/datasets/datasets_final/for_1_level/train_refactored_lematize_cut_final.csv",
#     dtype={'RGNTI1': str, 'RGNTI2': str, 'RGNTI3': str})
#
# df_level_2 = pd.read_csv("/Users/denismazepa/Desktop/Py_projects/VKR/datasets/datasets_final/for_other_levels/train_refactored_lematize_2_3_level.csv", dtype={'RGNTI1': str, 'RGNTI2': str, 'RGNTI3': str})
# # print(df_level_1["RGNTI1"].unique().)
# # print(df_level_2["RGNTI1"].unique().tolist())
# print(df_level_1["RGNTI1"].unique().tolist() == df_level_2["RGNTI1"].unique().tolist())
# print(len(df_level_2["RGNTI2"].unique().tolist()))
# print(df_level_2["RGNTI1"].unique().tolist())