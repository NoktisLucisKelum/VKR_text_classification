import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import pymorphy3
import nltk
import string

nltk.download('stopwords')

russian_stopwords = set(stopwords.words('russian'))


def remove_stopwords(text):
    # Разделяем текст на слова
    words = text.split()
    # Фильтруем слова, исключая стоп-слова
    filtered_words = [word.lower() for word in words if word.lower() not in russian_stopwords]
    # Объединяем слова обратно в строку
    return ' '.join(filtered_words)


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


class TextPreprocessor:
    def __init__(self, df):
        self.df = df
        self.morph = pymorphy3.MorphAnalyzer()
        self.stemmer = SnowballStemmer("russian")
        self.stop_words = set(stopwords.words("russian"))

    # def improve_columns(self, column_names):
    #     self.df = self.df.iloc[:, 0].str.split('\t', expand=True)
    #
    #     # Присваиваем имена столбцам
    #     self.df.columns = column_names
    #     self.df = self.df[column_names]

    def remove_punctuation(self, columns: list):
        """Удаляет пунктуация в нужных столбцах"""
        for column in columns:
            self.df[column] = self.df[column].apply(remove_punctuation)
            # self.df[column] = self.df[column].fillna('').astype(str).apply(lambda x: re.sub(r'[^\w\s]', '', x).lower())

    def lemmatize(self, columns: list):
        """Лемматизация"""
        for column in columns:
            self.df[column] = self.df[column].apply(
                lambda x: ' '.join([self.morph.parse(word)[0].normal_form for word in x.split()])
            )

    def stem(self, columns: list):
        """Стеммизация"""
        for column in columns:
            self.df[column] = self.df[column].apply(
                lambda x: ' '.join([self.stemmer.stem(word) for word in x.split()])
            )

    def remove_stop_words(self, columns: list):
        """Удаление стоп слов"""
        for column in columns:
            self.df[column] = self.df[column].apply(remove_stopwords)

    def convert_to_word_list(self, columns: list):
        """Перевод строки в массив строк"""
        for column in columns:
            self.df[column] = self.df[column].apply(lambda x: x.split())

    def save_to_csv(self, file_name: str):
        """Сохранение в файл"""
        self.df.to_csv(file_name, index=False)

    def drop_nan(self):
        """Удаление поломанных строк"""
        self.df = self.df.dropna()


# def split_and_rename_columns(dataframe: pd.DataFrame, column_names: list) -> pd.DataFrame:
#     """
#     Делит каждый DataFrame в списке по символу \t и присваивает имена столбцам.
#
#     :param dataframes: список pandas DataFrame, каждый с одной колонкой
#     :param column_names: список из шести строк, представляющих имена для новых столбцов
#     :return: список DataFrame, где каждая строка разделена на шесть столбцов с заданными именами
#     """
#
#     # Разделяем столбец по символу \t
#     split_df = dataframe.iloc[:, 0].str.split('\t', expand=True)
#
#     # Присваиваем имена столбцам
#     split_df.columns = column_names
#     split_df = split_df[column_names]
#
#     return split_df


df = pd.read_csv("datasets/datatsets_from_git/train/train_ru_work.csv", on_bad_lines='skip', sep='\t').head(400)
print(df.columns)
preprocessor = TextPreprocessor(df)
print(1)

# Последовательно применяем функции
preprocessor.drop_nan()
preprocessor.lemmatize(['title', 'body', 'keywords'])
preprocessor.remove_punctuation(['title', 'body'])
# # preprocessor.stem(['text_column'])
print(3)
preprocessor.remove_stop_words(['title', 'body'])
preprocessor.convert_to_word_list(['title', 'body', 'keywords'])
print(4)
preprocessor.save_to_csv("datasets/datasets_final/train_refactored_lematize.csv")

# print(preprocessor.df)
