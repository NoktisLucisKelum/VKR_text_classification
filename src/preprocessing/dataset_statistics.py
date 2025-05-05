import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer


class DataAnalyzer:
    def __init__(self, df):
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained('sergeyzh/rubert-tiny-turbo')

    def plot_bar_chart_frequency(self, column_name: str) -> None:
        """Подсчитываем встречаемость каждого числа"""
        self.df = self.df[~self.df["RGNTI1"].isin(['59', '86', '58', '0'])]
        value_counts = self.df[column_name].value_counts()

        plt.figure(figsize=(10, 6))
        plt.bar(value_counts.index.astype(str), value_counts.values)

        plt.xlabel("RGNTI1") # Используем имя столбца для метки оси X
        plt.ylabel('Частота')
        plt.title('Частота встречаемости в тренировочном наборе, уровень 1')

        plt.show()
        
    def unique_counts_level_1(self):
        """Выводит встречаемость уникальных строк в столбце RGNTI_1."""
        print(self.df["RGNTI1"].value_counts())
        print(self.df["RGNTI1"].unique().tolist())

    def unique_counts_level_2(self, column_name_value: str):
        """Выводит встречаемость уникальных строк в столбце RGNTI_2."""
        new_df = self.df[self.df["RGNTI1"] == column_name_value]
        print(new_df["RGNTI2"].value_counts())

    def unique_counts_level_3(self, column_name_1_value: str, column_name_2_value: str):
        """Выводит встречаемость уникальных строк в столбце RGNTI_3."""
        new_df = self.df[self.df["RGNTI1"] == column_name_1_value]
        new_df = new_df[new_df["RGNTI2"] == column_name_2_value]
        print(new_df["RGNTI3"].value_counts())

    def analyze_tokens(self, dataset_name: str):
        """Анализирует длину токенизированных текстов."""
        print(dataset_name)

        def get_tokenized_length(text):
            """Токенизирует текст и возвращает его длину."""
            tokens = self.tokenizer.encode(text, add_special_tokens=True) # Используем self.tokenizer
            return len(tokens)

        self.df['tokenized_length'] = self.df['body'].apply(get_tokenized_length)

        plt.figure(figsize=(10, 6))
        plt.hist(self.df['tokenized_length'], bins=50, color='blue', alpha=0.7)
        plt.title(f'Распределение токенов в тренировчоном датасете, rubert-tiny-turbo')
        plt.xlabel('Длина токенов')
        plt.ylabel('Количество')
        plt.show()

        percentile_95 = self.df['tokenized_length'].quantile(0.95)
        percentile_99 = self.df['tokenized_length'].quantile(0.99)

        print(f"95th percentile length: {percentile_95}")
        print(f"99th percentile length: {percentile_99}")

    def word_counting(self, dataset_name: str) -> None:
        """Записывает среднее количество слов в тексте"""

        def word_count(text):
            return len(text.split())

        self.df['word_count'] = self.df['body'].apply(word_count)
        average_word_count = self.df['word_count'].mean()
        print(f'Средняя длина текстов в словах в {dataset_name}: {average_word_count}')

    def analis_labels(self, column_1: str, column_2: str) -> None:
        """Количество уникальных значений в колонке 2 для каждого значения в колонке 1 в среднем"""
        print(self.df.groupby(column_1)[column_2].nunique().mean())

    def analis_len_labels(self, column_1: str, column_2: str) -> None:
        """Количество уникальных значений в колонке 2 для каждого значения в колонке 1"""
        unique_counts = self.df.groupby(column_1)[column_2].nunique()
        print(unique_counts.reset_index(name='unique_count'))

    def pict_label_analys(self, col1: str, col2: str):
        unique_counts = self.df.groupby(col1)[col2].nunique()
        frequency = unique_counts.value_counts().sort_index()

        #
        plt.figure(figsize=(10, 6))
        plt.bar(frequency.index, frequency.values, color='skyblue', edgecolor='black')
        plt.xlabel(f"Количество уникальных значений в '{col2}' для каждого значения '{col1}'", fontsize=12)
        plt.ylabel("Частота", fontsize=12)
        plt.title("График: Количество уникальных значений vs Частота", fontsize=14)
        plt.xticks(frequency.index, fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()


# df = pd.read_csv(
#     '/datasets/datasets_final/for_1_level/test_refactored_lematize_cut_final.csv')
df_2 = pd.read_csv(
    '/Users/denismazepa/Desktop/Py_projects/VKR/src/preprocessing/train_big_augmented_uncut_preprocessed_final_level_1.csv')

statisticer = DataAnalyzer(df_2)
# statisticer.analyze_tokens("Тренировочный датасет")
# analyze_tokens(df, "Test dataset")
# statisticer.word_counting("Train dataset")
# word_counting(df, "Test dataset")
# statisticer.analis_labels("RGNTI2", "RGNTI3")
# statisticer.analis_len_labels("RGNTI2", "RGNTI3")
# statisticer.pict_label_analys("RGNTI2", "RGNTI3")
statisticer.plot_bar_chart_frequency("RGNTI1")
