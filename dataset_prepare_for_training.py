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

nltk.download('stopwords')

rdkit.RDLogger.DisableLog('rdApp.*')
rdkit.RDLogger.EnableLog('rdApp.*')

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
        return f" {"ывжифлспдлиыво"} {formula} {"ывжифлспдлиыво"} "
    return re.sub(pattern,  replace_formula, text)


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

    def drop_columns(self, columns_to_drop: list):
        self.df = self.df.drop(columns=columns_to_drop)

    def remove_english_strings(self):
        self.df['latin_ratio'] = self.df['body'].apply(
            lambda s: sum(c.isalpha() and c.lower() in 'abcdefghijklmnopqrstuvwxyz' for c in s) / len(s) if isinstance(
                s, str) else 0
        )
        self.df = self.df[self.df['latin_ratio'] <= 0.6].drop('latin_ratio', axis=1)


df = pd.read_csv("datasets/datatsets_from_git/train/train_ru_work.csv",
                 on_bad_lines='skip', sep='\t').head(10000)
# print(df.columns)
preprocessor = TextPreprocessor(df)
print(1)

# Последовательно применяем функции
preprocessor.drop_nan()
preprocessor.chem_formula_prepare()
preprocessor.phys_formula_prepare()
preprocessor.remove_english_strings()
preprocessor.lemmatize(['title', 'body', 'keywords'])
preprocessor.remove_punctuation(['title', 'body'])
# # preprocessor.stem(['text_column'])
preprocessor.remove_stop_words(['title', 'body'])
preprocessor.convert_to_word_list(['title', 'body', 'keywords'])
preprocessor.remove_second_index(["RGNTI1", "RGNTI2", "RGNTI3"])
preprocessor.drop_columns(["correct"])
preprocessor.save_to_csv("datasets/datasets_final/train_refactored_lematize.csv")

# print(preprocessor.df)
