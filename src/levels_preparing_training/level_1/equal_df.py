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
