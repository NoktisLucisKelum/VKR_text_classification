import pandas as pd
#
# # df = pd.read_csv("VKR/datasets/datasets_final/test_refactored_lematize.csv")
# df = pd.read_csv("/Users/denismazepa/Desktop/Py_projects/VKR/datasets/datasets_final/test_refactored_lematize.csv")
# print(len(df), type(df["RGNTI1"]))

import pandas as pd


class DataFrameProcessor:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def limit_unique_rows(self, column_name, unique_value, n):
        """Оставляет только n строк для заданного уникального значения в заданном столбце."""
        if column_name in self.dataframe.columns:
            # Фильтруем строки с указанным значением
            filtered_df = self.dataframe[self.dataframe[column_name] == unique_value]
            # Оставляем только первые n строк
            limited_df = filtered_df.head(n)
            # Оставляем остальные строки
            remaining_df = self.dataframe[self.dataframe[column_name] != unique_value]
            # Объединяем обратно
            self.dataframe = pd.concat([limited_df, remaining_df], ignore_index=True)
        else:
            raise ValueError(f"Столбец '{column_name}' не найден в DataFrame.")

# Пример использования
df = pd.DataFrame({
    'A': [1, 2, 2, 3, 3, 3],
    'B': ['x', 'y', 'z', 'w', 'v', 'u']
})

processor = DataFrameProcessor(df)
processor.limit_unique_rows('A', 3, 2)
print(processor.dataframe)