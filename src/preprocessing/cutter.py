import pandas as pd


class DataFrameProcessor:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def save_dataset(self, file_path):
        """Сохраняет DataFrame в файл CSV."""
        self.dataframe.to_csv(file_path, index=False)

    def unique_counts_level_1(self):
        """Выводит встречаемость уникальных строк в заданном столбце."""
        print(self.dataframe["RGNTI1"].value_counts())

    def unique_counts_level_2(self, column_name_value: str):
        """Выводит встречаемость уникальных строк в заданном столбце."""
        new_df = self.dataframe[self.dataframe["RGNTI1"] == column_name_value]
        print(new_df["RGNTI2"].value_counts())

    def unique_counts_level_3(self, column_name_1_value: str, column_name_2_value: str):
        new_df = self.dataframe[self.dataframe["RGNTI1"] == column_name_1_value]
        new_df = new_df[new_df["RGNTI2"] == column_name_2_value]
        print(new_df["RGNTI3"].value_counts())

    def delete_strings(self, list_of_values):
        self.dataframe = self.dataframe[~self.dataframe["RGNTI1"].isin(list_of_values)]

    def limit_unique_rows(self, column_name, unique_value, n):
        """Оставляет только n строк для заданного уникального значения в заданном столбце."""
        if column_name in self.dataframe.columns:
            # Фильтруем строки с указанным значением
            filtered_df = self.dataframe[self.dataframe[column_name] == unique_value]
            print(len(filtered_df))
            # Оставляем только первые n строк
            limited_df = filtered_df.head(n)
            # Оставляем остальные строки
            remaining_df = self.dataframe[self.dataframe[column_name] != unique_value]
            # Объединяем обратно
            self.dataframe = pd.concat([limited_df, remaining_df], ignore_index=True)
        else:
            raise ValueError(f"Столбец '{column_name}' не найден в DataFrame.")


df = pd.read_csv('/Users/denismazepa/Desktop/Py_projects/VKR/datasets/datasets_final'
                 '/test_refactored_lematize.csv', dtype={'RGNTI1': str, 'RGNTI2': str, 'RGNTI3': str})
# print(df["RGNTI1"].dtype)
# print(df[df['RGNTI1'] == '58'])
preprocessor = DataFrameProcessor(df)
preprocessor.unique_counts_level_1()
preprocessor.delete_strings(['0', '59', '58', '86', '00'])
preprocessor.limit_unique_rows("RGNTI1", "34", 17000)
preprocessor.unique_counts_level_1()
preprocessor.save_dataset('/Users/denismazepa/Desktop/Py_projects/VKR/datasets/datasets_final/test_refactored_lematize_cut_final.csv')
