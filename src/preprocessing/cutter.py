import pandas
import pandas as pd


class DataFrameProcessor:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def save_dataset(self, file_path):
        """Сохраняет DataFrame в файл CSV."""
        self.dataframe.to_csv(file_path, index=False)

    def unique_counts_level_1(self):
        """Выводит встречаемость уникальных строк в столбце RGNTI_1."""
        print(self.dataframe["RGNTI1"].value_counts())
        print(self.dataframe["RGNTI1"].unique().tolist())

    def unique_counts_level_2(self, column_name_value: str):
        """Выводит встречаемость уникальных строк в столбце RGNTI_2."""
        new_df = self.dataframe[self.dataframe["RGNTI1"] == column_name_value]
        print(new_df["RGNTI2"].value_counts())

    def unique_counts_level_3(self, column_name_1_value: str, column_name_2_value: str):
        """Выводит встречаемость уникальных строк в столбце RGNTI_3."""
        new_df = self.dataframe[self.dataframe["RGNTI1"] == column_name_1_value]
        new_df = new_df[new_df["RGNTI2"] == column_name_2_value]
        print(new_df["RGNTI3"].value_counts())

    def delete_strings(self, list_of_values: list):
        """Удаляет строки где в столбце RGNTI_1 встречаются значения из списка"""
        self.dataframe = self.dataframe[~self.dataframe["RGNTI1"].isin(list_of_values)]

    def limit_unique_rows(self, column_name, unique_value, n: int):
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

    def pad_single_char_values(self, column_name):
        """Добавляет '0' перед строками с одним символом в указанном столбце."""
        self.dataframe[column_name] = self.dataframe[column_name].apply(
            lambda x: '0' + x if isinstance(x, str) and len(x) == 1 else x
        )

    def pad_single_char_values_reverse(self, column_name):
        """Добавляет '0' перед и после строками с четырьмя символами в указанном столбце."""
        self.dataframe[column_name] = self.dataframe[column_name].apply(
            lambda x: x + '0' if isinstance(x, str) and len(x) == 4 and str(x)[2] == '.' else x)
        self.dataframe[column_name] = self.dataframe[column_name].apply(
            lambda x: '0' + x if isinstance(x, str) and len(x) == 4 and str(x)[1] == '.' else x)

    def return_dataset(self):
        return self.dataframe


def limit_unique_values(df: pandas.DataFrame, column_name: str, n=200) -> pd.DataFrame:
    """Обрезает все классы в RGNTI_1 до n с начала"""
    result_df = pd.DataFrame()

    for value in df[column_name].unique():
        subset = df[df[column_name] == value]
        if len(subset) > n:
            subset = subset.head(n)
        result_df = pd.concat([result_df, subset])

    return result_df


def limit_unique_values_tail(df: pandas.DataFrame, column_name: str, n=40) -> pd.DataFrame:
    """Обрезает все классы в RGNTI_1 до n с конца"""
    result_df = pd.DataFrame()

    for value in df[column_name].unique():
        subset = df[df[column_name] == value]
        if len(subset) > n:
            subset = subset.tail(n)
        result_df = pd.concat([result_df, subset])

    return result_df


# df_null = limit_unique_values_tail(pd.read_csv('/Users/denismazepa/Desktop/Py_projects/VKR/datasets/datasets_final'
#                       '/test_refactored_lematize.csv', dtype={'RGNTI1': str, 'RGNTI2': str, 'RGNTI3': str}), 'RGNTI1')
# proc_null = DataFrameProcessor(df_null)
# proc_null.pad_single_char_values('RGNTI1')
# proc_null.delete_strings(['0', '59', '58', '86', '00'])
#
# proc_null.save_dataset('/Users/denismazepa/Desktop/Py_projects/VKR/datasets/datasets_small/train_refactored_small_validation.csv')
#
# df_zero = limit_unique_values(pd.read_csv('/Users/denismazepa/Desktop/Py_projects/VKR/datasets/datasets_final'
#                       '/test_refactored_lematize.csv', dtype={'RGNTI1': str, 'RGNTI2': str, 'RGNTI3': str}), 'RGNTI1')
# df_one = limit_unique_values(pd.read_csv('/Users/denismazepa/Desktop/Py_projects/VKR/datasets/datasets_final'
#                   '/train_refactored_lematize.csv', dtype={'RGNTI1': str, 'RGNTI2': str, 'RGNTI3': str}), 'RGNTI1')
#
# proc_zero = DataFrameProcessor(df_zero)
# proc_zero.pad_single_char_values('RGNTI1')
# proc_zero.delete_strings(['59', '58', '86', '00'])
#
# proc_one = DataFrameProcessor(df_one)
# proc_one.pad_single_char_values('RGNTI1')
# proc_one.delete_strings(['59', '58', '86', '00'])
#
# proc_one.save_dataset('/Users/denismazepa/Desktop/Py_projects/VKR/datasets/datasets_small/train_refactored_small.csv')
# proc_zero.save_dataset('/Users/denismazepa/Desktop/Py_projects/VKR/datasets/datasets_small/test_refactored_small.csv')
"""__________________________________"""
df = pd.read_csv('../../datasets/datasets_final/for_1_level/test_refactored_lematize_no_numbers.csv', dtype={'RGNTI1': str, 'RGNTI2': str, 'RGNTI3': str})
# print(df["RGNTI1"].dtype)
# print(df[df['RGNTI1'] == '58'])
preprocessor_1 = DataFrameProcessor(df)
preprocessor_1.unique_counts_level_1()
preprocessor_1.delete_strings(['0', '59', '58', '86', '00'])
# preprocessor_1.limit_unique_rows("RGNTI1", "34", 17000)
preprocessor_1.unique_counts_level_1()
preprocessor_1.pad_single_char_values('RGNTI1')
preprocessor_1.pad_single_char_values_reverse('RGNTI2')
preprocessor_1.save_dataset('test_refactored_lematize_no_numbers_2_3_level.csv')


df_new = pd.read_csv('../../datasets/datasets_final/for_1_level/train_refactored_lematize_no_numbers.csv', dtype={'RGNTI1': str, 'RGNTI2': str, 'RGNTI3': str})
# print(df["RGNTI1"].dtype)
# print(df[df['RGNTI1'] == '58'])
preprocessor = DataFrameProcessor(df_new)
preprocessor.unique_counts_level_1()
preprocessor.delete_strings(['0', '59', '58', '86', '00'])
# preprocessor.limit_unique_rows("RGNTI1", "34", 28000)
preprocessor.unique_counts_level_1()
preprocessor.pad_single_char_values('RGNTI1')
preprocessor.pad_single_char_values_reverse('RGNTI2')
preprocessor.save_dataset('train_refactored_lematize_no_numbers_2_3_level.csv')
