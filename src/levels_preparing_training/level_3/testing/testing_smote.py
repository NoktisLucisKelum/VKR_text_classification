import os
import pandas as pd
from imblearn.pipeline import Pipeline
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import json
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from time import time
from src.preprocessing.dataset_prepare_for_training import (TextPreprocessor)
import warnings
from imblearn.over_sampling import SMOTE, SMOTEN
from imblearn.over_sampling import RandomOverSampler



df = pd.read_csv(
    "/Users/denismazepa/Desktop/Py_projects/VKR/datasets/datasets_final/for_other_levels/train_refactored_lematize_no_numbers_2_3_level.csv",
    dtype={'RGNTI1': str, 'RGNTI2': str, 'RGNTI3': str})

df_1 = df[(df['RGNTI1'] == '34') & (df['RGNTI2'] == '34.39')]
df_2 = df[(df['RGNTI1'] == '41') & (df['RGNTI2'] == '41.51')]
df_3 = df[(df['RGNTI1'] == '55') & (df['RGNTI2'] == '55.39')]
df_4 = df[(df['RGNTI1'] == '50') & (df['RGNTI2'] == '50.01')]
df_5 = df[(df['RGNTI1'] == '29') & (df['RGNTI2'] == '29.27')]
df_6 = df[(df['RGNTI1'] == '73') & (df['RGNTI2'] == '73.34')]
df_7 = df[(df['RGNTI1'] == '38') & (df['RGNTI2'] == '38.61')]
df_8 = df[(df['RGNTI1'] == '37') & (df['RGNTI2'] == '37.15')]
df_9 = df[(df['RGNTI1'] == '28') & (df['RGNTI2'] == '28.23')]
df_10 = df[(df['RGNTI1'] == '61') & (df['RGNTI2'] == '61.67')]
df_11 = df[(df['RGNTI1'] == '61') & (df['RGNTI2'] == '61.39')]
df_12 = df[(df['RGNTI1'] == '37') & (df['RGNTI2'] == "37.25")]
df_13 = df[(df['RGNTI1'] == '87') & (df['RGNTI2'] == '87.03')]
df_14 = df[(df['RGNTI1'] == '47') & (df['RGNTI2'] == '47.09')]
df_15 = df[(df['RGNTI1'] == '68') & (df['RGNTI2'] == '68.03')]

dict_of_frames = {"1": [df_1, "Физиология человека и животных"],
                  "2": [df_2, "Обсерватории. Инструменты, приборы и методы астрономических наблюдений"],
                  "3": [df_3, "Химическое и нефтяное машиностроение"],
                  "4": [df_4, "Общие вопросы автоматики и вычислительной техники"],
                  "5": [df_5, "Физика плазмы"], "6": [df_6, "Водный транспорт"], "7": [df_7, "Гидрогеология"],
                  "8": [df_8, "Геомагнетизм и высокие слои атмосферы"], "9": [df_9, "Искусственный интеллект"],
                  "10": [df_10, "Технология химических волокон и нитей"],
                  "11": [df_11, "Промышленный синтез органических красителей и пигментов"],
                  "12": [df_12, "Океанология"],
                  "13": [df_13, "Теория и методы изучения и охраны окружающей среды. "
                                "Экологические основы использования природных ресурсов"],
                  "14": [df_14, "Материалы для электроники и радиотехники"],
                  "15": [df_15, "Сельскохозяйственная биология"]}


pipeline_0 = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.5, ngram_range=(1, 1), use_idf=True)),
    ('svc', LinearSVC(C=0.65, class_weight='balanced', fit_intercept=False, loss='squared_hinge', max_iter=4000,
                      penalty='l2', tol=0.0001))
])
pipeline_smote = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.5, ngram_range=(1, 1), use_idf=True)),
    ('smote', SMOTE(random_state=42, sampling_strategy='auto', k_neighbors=2)),
    ('svc', LinearSVC(C=0.65, class_weight='balanced', fit_intercept=False, loss='squared_hinge', max_iter=4000,
                      penalty='l2', tol=0.0001))
])

pipeline_smoten = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.5, ngram_range=(1, 1), use_idf=True)),
    ('smoten', SMOTEN(random_state=42, sampling_strategy='auto')),
    ('svc', LinearSVC(C=0.65, class_weight='balanced', fit_intercept=False, loss='squared_hinge', max_iter=4000,
                      penalty='l2', tol=0.0001))
])


for i in dict_of_frames.keys():
    X_train, X_test, y_train, y_test = train_test_split(dict_of_frames[i][0]['body'],
                                                        dict_of_frames[i][0]['RGNTI3'],
                                                        test_size=0.2, random_state=42)
    pipeline_0.fit(X_train, y_train)
    y_pred = pipeline_0.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(
        f'Без SMOTE, Датасет: {i}, Векторизатор: TF-IDF, F1 Score: {f1:.3f}, Длина датасета: {len(dict_of_frames[i][0])},'
        f'Количество классов в датасете:  {len(dict_of_frames[i][0]['RGNTI3'].unique().tolist())},'
        f'  Тема: {dict_of_frames[i][1]}')

for i in dict_of_frames.keys():
    X_train, X_test, y_train, y_test = train_test_split(dict_of_frames[i][0]['body'],
                                                        dict_of_frames[i][0]['RGNTI3'],
                                                        test_size=0.2, random_state=42)
    pipeline_smote.fit(X_train, y_train)
    y_pred = pipeline_smote.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(
        f'SMOTE, Датасет: {i}, Векторизатор: TF-IDF, F1 Score: {f1:.3f}, Длина датасета: {len(dict_of_frames[i][0])},'
        f'Количество классов в датасете:  {len(dict_of_frames[i][0]['RGNTI3'].unique().tolist())},'
        f'  Тема: {dict_of_frames[i][1]}')

for i in dict_of_frames.keys():
    X_train, X_test, y_train, y_test = train_test_split(dict_of_frames[i][0]['body'],
                                                        dict_of_frames[i][0]['RGNTI3'],
                                                        test_size=0.2, random_state=42)
    pipeline_smoten.fit(X_train, y_train)
    y_pred = pipeline_smoten.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(
        f'SMOTEN, Датасет: {i}, Векторизатор: TF-IDF, F1 Score: {f1:.3f}, Длина датасета: {len(dict_of_frames[i][0])},'
        f'Количество классов в датасете:  {len(dict_of_frames[i][0]['RGNTI3'].unique().tolist())},'
        f'  Тема: {dict_of_frames[i][1]}')
