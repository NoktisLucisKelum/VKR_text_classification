import pandas as pd


def filter_by_column(df, output_file, column_name, threshold=30):
    value_counts = df[column_name].value_counts()
    common_values = value_counts[value_counts >= threshold].index
    filtered_df = df[df[column_name].isin(common_values)]

    filtered_df.to_csv(output_file, index=False)
    print(f"Размер после фильтрации по столбцу '{column_name}': {len(filtered_df)} строк")
    # return filtered_df


filter_by_column("train_big_augmented_uncut_preprocessed_final.csv",
                 "train_big_augmented_uncut_final_level_2_3.csv", "RGNTI2")