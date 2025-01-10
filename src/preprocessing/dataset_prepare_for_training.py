import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import pymorphy3
import nltk
import string
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
import re
import rdkit
import difflib

nltk.download('stopwords')

from rdkit import rdBase
blocker = rdBase.BlockLogs()

russian_stopwords = set(stopwords.words('russian'))


def remove_stopwords(text: str):
    """Убираем ненужные слова и сводим к нижнему регистру"""
    words = text.split()
    # Фильтруем слова, исключая стоп-слова
    filtered_words = [word.lower() for word in words if word.lower() not in russian_stopwords]
    # Объединяем слова обратно в строку
    return ' '.join(filtered_words)


def remove_punctuation(text: str) -> str:
    """Удаляет знаки препинания из текста, кроме тех, что находятся внутри [CHEM]...[/CHEM]."""

    def replace_punctuation(match):
        return match.group(0).translate(str.maketrans('', '', string.punctuation))

    return re.sub(r'(\[chem].*?\[/chem])|[\W_]+', lambda m: m.group(1) or replace_punctuation(m), text)


def standardize_chemical_formula(text):
    """
    Standardizes chemical formulas in text using RDKit.
    Formulas will be enclosed in [CHEM]...[/CHEM] tags.
    """

    def replace_formula(match):
        formula = match.group(0)
        try:
            mol = Chem.MolFromSmiles(formula)
            if mol:
                mol = Chem.AddHs(mol)  # Add hydrogens for better depiction
                rdMolDraw2D.PrepareMolForDrawing(mol)

                formula = Chem.MolToSmiles(mol, isomericSmiles=False)
                return f"[CHEM]{formula}[/CHEM]"
        except:
            pass
        return f"[CHEM]{formula}[/CHEM]"

    return re.sub(r'([A-Z][a-z]?\d*(\(|\[|\{).*?(\)|\]|\})|[A-Z][a-z]?\d*[^A-Za-z\s]+)', replace_formula, text)


def mark_formula_with_text(text):
    pattern = r'[A-Za-z]+(?:_[A-Za-z]+)?\s*(?:[=~]+|\s*≈\s*)\s*[^\s]+'

    def replace_formula(match):
        formula = match.group(0)
        return f" {"ывжифлспдлиыво"}{formula}{"ывжифлспдлиыво"} "
    return re.sub(pattern,  replace_formula, text)


def split_words(word_list):
    result = []
    for word in word_list.split():
        result.extend(word.split('\\'))  # Разделяем слова по символу '\'
    return ' '.join(result)


def highlight_similar_words(text, reference, threshold=0.9):
    words_list = text.split()  # Разбиваем строку на слова
    highlighted_words = []
    reference_list = reference.split(' ')
    for word in words_list:
        # Находим наиболее похожее слово из reference_list
        matches = difflib.get_close_matches(word, reference_list, n=1, cutoff=threshold)
        if matches:
            # Если есть совпадение, оборачиваем слово в 'jfisblaku vhsljdka'
            highlighted_words.append(f'jfisblaku_{word}_vhsljdka')
        else:
            highlighted_words.append(word)
    return ' '.join(highlighted_words)  # Объединяем слова обратно в строку


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

    def chem_formula_prepare(self):
        """Работает с химическими формулами"""
        self.df['body'] = self.df['body'].apply(standardize_chemical_formula)

    def phys_formula_prepare(self):
        """Работает с физическими формулами"""
        self.df["body"] = self.df["body"].apply(mark_formula_with_text)

    def remove_punctuation(self, columns: list):
        """Удаляет пунктуация в нужных столбцах"""
        for column in columns:
            self.df[column] = self.df[column].apply(remove_punctuation)
            # self.df[column] = self.df[column].fillna('').astype(str).apply(lambda x: re.sub(r'[^\w\s]', '', x).lower())

    def remove_second_index(self, columns: list):
        for column in columns:
            self.df[column] = self.df[column].str.split('\\', n=1).str[0]

    def lemmatize(self, columns: list):
        """Лемматизация"""
        for column in columns:
            self.df[column] = self.df[column].apply(
                lambda x: ' '.join([self.morph.parse(word)[0].normal_form for word in x.split()])
                if isinstance(x, str) else x)

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

    def drop_columns(self, columns_to_drop: list):
        """Удаление столбцов"""
        self.df = self.df.drop(columns=columns_to_drop)

    def remove_english_strings(self):
        """Удаление строк где моного латинских букв"""
        self.df['latin_ratio'] = self.df['body'].apply(
            lambda s: sum(c.isalpha() and c.lower() in 'abcdefghijklmnopqrstuvwxyz' for c in s) / len(s) if isinstance(
                s, str) else 0
        )
        self.df = self.df[self.df['latin_ratio'] <= 0.6].drop('latin_ratio', axis=1)

    def merge_and_drop_columns(self, column1: str, column2: str):
        """Соединяет тело статьи и название"""
        self.df[column1] = self.df[column1] + self.df[column2]
        self.df.drop(column2, axis=1, inplace=True)

    def repare_columns(self):
        """делит ключевые слова на двое"""
        self.df['keywords'] = self.df['keywords'].apply(split_words)
        self.df['body'] = self.df.apply(lambda row: highlight_similar_words(row['body'], row['keywords']), axis=1)

    # def refactor(self):
    #     """Превращает столбцы int в str"""
    #     self.df["RGNTI1"] = self.df["RGNTI1"].apply(str)
    #     self.df["RGNTI2"] = self.df["RGNTI2"].apply(str)
    #     self.df["RGNTI3"] = self.df["RGNTI3"].apply(str)

    def printing(self, column: str):
        print(type(self.df[column]), self.df[column].head(10))

    def astp(self, column: str):
        """Выводит тип колонки"""
        print(self.df[column].dtype)

    def return_dataset(self):
        return self.df

