import torch
import pandas as pd
from time import time
from transformers import T5ForConditionalGeneration, T5Tokenizer

MODEL_NAME = 'cointegrated/rut5-base-paraphraser'
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def paraphrase(text, beams=9, grams=3):
    x = tokenizer(text, return_tensors='pt', padding=True).to(model.device)
    max_size = int(x.input_ids.shape[1] * 1.5 + 10)
    out = model.generate(**x, encoder_no_repeat_ngram_size=grams, num_beams=beams, max_length=max_size)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def augment_with_paraphrases(dataset: pd.DataFrame, target_values: list, check_column: str = "RGNTI1",
                             text_column: str = "body") -> pd.DataFrame:
    print(f"Изначальный датасет: {len(dataset)}")
    """
    Аугментирует датасет перефразированными текстами для строк с указанными значениями в целевой колонке.

    Параметры:
    - dataset: исходный DataFrame
    - target_values: список значений в колонке check_column, для которых нужно делать аугментацию
    - check_column: название колонки для проверки значений (по умолчанию "RGNTI1")
    - text_column: название колонки с текстом для перефразирования (по умолчанию "body")
    - paraphrase_func: функция для перефразирования текста (должна принимать строку и возвращать строку)

    Возвращает:
    - Новый DataFrame с добавленными перефразированными строками
    """


    # Создаем копию датасета, чтобы не изменять исходный
    result_df = dataset.copy()
    new_rows = []

    # Находим строки, где значение RGNTI1 входит в target_values
    mask = dataset[check_column].isin(target_values)
    target_rows = dataset[mask]
    cnt = 0

    # Проходим по всем подходящим строкам
    for _, row in target_rows.iterrows():
        cnt += 1
        try:
            paraphrased_text = paraphrase(row[text_column])
            if cnt % 75 == 4:
                now = time()
                print(f"Изначальный текст: {row[text_column]}")
                print(f"Перефразированный текст: {paraphrased_text}")
                print(time() - now)

            # Создаем новую строку с перефразированным текстом
            new_row = row.copy()
            new_row[text_column] = paraphrased_text
            new_rows.append(new_row)

        except Exception as e:
            print(f"Ошибка при перефразировании текста: {row[text_column]}")
            print(f"Ошибка: {e}")
            continue

    # Если есть новые строки, добавляем их к результату
    if new_rows:
        new_rows_df = pd.DataFrame(new_rows)
        result_df = pd.concat([result_df, new_rows_df], ignore_index=True)
    print(f"Конечный датасет: {len(result_df)}")

    return result_df


from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import pymorphy3
import nltk
import string
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
import re
import difflib
from rdkit import rdBase

nltk.download('stopwords')

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
        return f" {"physmath"}{formula}{"physmath"} "
    return re.sub(pattern,  replace_formula, text)


def split_words(word_list):
    result = []
    if not(isinstance(word_list, str)):
        word_list = str(word_list)
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
            highlighted_words.append(f'special_{word}_special')
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

    def split_column_value(self, column_name: str, new_column_name: str = 'new_column'):
        """
        Принимает на вход DataFrame и название столбца.
        Из значения столбца выделяет часть строки после символа '/' и
        записывает её в новый столбец, который по умолчанию называется 'new_column'.
        Возвращает изменённый DataFrame.
        """

        def get_second_part(value):
            if pd.isna(value):
                return None
            parts = value.split('\\')
            if len(parts) > 1:
                return parts[1]
            return parts[0]

        # Применяем функцию к нужному столбцу и сохраняем результат в новый столбец
        self.df[new_column_name] = self.df[column_name].apply(get_second_part)

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
        self.df = self.df[self.df['latin_ratio'] <= 0.65].drop('latin_ratio', axis=1)

    def remove_numbers(self):
        self.df['body'] = self.df['body'].str.replace(r'\d+', '', regex=True)

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


df = pd.read_csv("/Users/denismazepa/Desktop/Py_projects/VKR/datasets/datasets_colide/train_mixed_data.csv",
                 sep="\t", on_bad_lines='skip')
# dfx = pd.read_csv("numbers.csv")
# print(dfx['keywords'])
# df_new = pd.read_csv("/Users/denismazepa/Desktop/Py_projects/VKR/datasets/datasets_final/for_1_level/train_refactored_lematize_no_numbers.csv")
# print(df_new.columns)
# print(df_new["body"].head())
# print(df_new["RGNTI1"].value_counts())
print(df.columns)
preprocessor = TextPreprocessor(df)

# Последовательно применяем функции
preprocessor.drop_nan()
print(1)
preprocessor.split_column_value('RGNTI1', 'RGNTI1_2')
print(7)
preprocessor.remove_second_index(['RGNTI1', "RGNTI2", "RGNTI3"])
print(8)
s = preprocessor.return_dataset()
s.drop(s[s["RGNTI1"].isin(["58", "00", "59", "86", "69", "12", "75", "67", "62", "19"])].index, inplace=True)
print(len(s["RGNTI1"].value_counts()))
print(2)
print(s["RGNTI1"].value_counts())
# preprocessor.remove_english_strings()
# s = preprocessor.return_dataset()
#
#
# # print(s, type(s))
# print(s["RGNTI1"].value_counts())
# new_df = augment_with_paraphrases(s, ["60", "64", "66", "19", "62", "67", "75", "12", "69"])
# preprocessor_new = TextPreprocessor(new_df)
# print(len(new_df["RGNTI1"].value_counts()))
# preprocessor_new.chem_formula_prepare()
# preprocessor_new.phys_formula_prepare()
# print(3)
# preprocessor_new.remove_numbers()
# print(3.5)
# preprocessor_new.lemmatize(['title', 'body', 'keywords'])
# print(4)
# preprocessor_new.remove_punctuation(['title', 'body'])
# print(5)
# # # preprocessor.stem(['text_column'])
# preprocessor_new.remove_stop_words(['title', 'body'])
# print(6)
# preprocessor_new.merge_and_drop_columns('body', 'title')
# print(10)
# # preprocessor.printing('body')
# # preprocessor.printing('keywords')
# preprocessor_new.repare_columns()
# preprocessor_new.drop_columns(["correct"])
# preprocessor_new.remove_second_index(["RGNTI1", "RGNTI2", "RGNTI3"])
# preprocessor_new.save_to_csv("train_full_uncut_final.csv")
#
#
# sx = preprocessor_new.return_dataset()
# print(type(sx), len(sx))
# print(sx.columns)
#
# #
# # df = pd.read_csv("_numbers.csv")
# # print(df.columns)
