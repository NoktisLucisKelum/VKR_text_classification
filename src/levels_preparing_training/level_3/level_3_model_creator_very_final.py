import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import json
import re

import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv(
    "/Users/denismazepa/Desktop/Py_projects/VKR/src/preprocessing/train_big_augmented_uncut_preprocessed_final_level_2_3.csv",
    dtype={'RGNTI1': str, 'RGNTI2': str, 'RGNTI3': str})
print(0)

with open("/Users/denismazepa/Desktop/Py_projects/VKR/grnti/GRNTI_2_ru.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

print(1)
# df_reserve = pd.read_csv(
#     "/Users/denismazepa/Desktop/Py_projects/VKR/datasets/datatsets_from_git/train/train_ru_work.csv",
#     sep="\t", on_bad_lines='skip')

columns = ["Index_name", "Name", "f1_weighted", "f1_micro", "f1_macro", "length", "unique_label", "Augmented"]
data_tos_save = pd.DataFrame(columns=columns)


pipeline_0 = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.5, ngram_range=(1, 1), use_idf=True)),
    ('svc', LinearSVC(C=0.65, class_weight='balanced', fit_intercept=False, loss='squared_hinge', max_iter=4000,
                      penalty='l2', tol=0.0001))
])
#
# list_of_before_forecasts = ["34.03", "34.15", "34.19", "34.21", "34.23", "34.25", "34.27",
#                             "34.29", "34.31", "34.333", "34.35", "34.37", "34.39", "34.41",
#                             "34.43", "34.45", "34.47", "34.49", "34.55", "47.05", "55.13",
#                             "87.29", "34.15", "53.49"]


list_of_unique = df["RGNTI2"].unique().tolist()
dict_of_success = {}
sums, count, count_high_f1 = 0, 0, 0
count_passed_one_label = 0
count_passed_small = 0
count_need_augmentation = 0
sum_real, count_real = 0, 0
count_no_name = 0

# print(list_of_unique)
for i in list_of_unique:
    print(i)
    try:
        model_name = data[i].replace(' ', '_')
    except Exception:
        model_name = f"Выборка_с_отсутствующим_названием_{count}"
        count_no_name += 1
    df_cut = df[df["RGNTI2"] == i]
    len_of_sample = len(df_cut)
    if len(df_cut["RGNTI3"].unique().tolist()) == 1:
        count_passed_one_label += 1
        print(f"Название темы: {model_name}, Код темы: {i}, Состоит из {len(df_cut["RGNTI3"].unique().tolist())} "
              f"класса, и имеет размер {len_of_sample}")
        data_tos_save = data_tos_save._append({
            "Index_name": i,
            "Name": model_name,
            "f1_weighted": 0,
            "f1_micro": 0,
            "f1_macro": 0,
            "length": len(df_cut),
            "unique_label": 1,
            "string_length_average": 0,
            "string_length_median": 0,
            'доля_спец_токенов': 0,
            'лекс_разнообразие': 0
        }, ignore_index=True)
        data_tos_save.to_excel("Results_level_3_big.xlsx")
    elif len_of_sample < 10:
        count_passed_small += 1
        print(f"Название темы: {model_name}, Код темы: {i}, Состоит из {len(df_cut["RGNTI3"].unique().tolist())} "
              f"класса, и имеет размер {len_of_sample}")
    else:
        X_train, X_test, y_train, y_test = train_test_split(df_cut['body'], df_cut['RGNTI3'],
                                                            test_size=0.2, random_state=42)
        pipeline = pipeline_0
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        f1_weighted = f1_score(y_test, y_pred, average="weighted")
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_micro = f1_score(y_test, y_pred, average="micro")
        if f1_micro == 1.00 or f1_macro == 1.00 or f1_weighted == 1.00:
            count_high_f1 += 1
        else:
            sum_real += f1_weighted
            count_real += 1


        sums += f1_weighted
        count += 1
        dict_of_success[i] = f1_weighted

        text_lengths = [len(str(t).split()) for t in df_cut['body']]
        special_tokens = sum(len(re.findall(r'\[[a-z]+\]', str(t))) for t in df_cut['body'])
        total_words = sum(len(str(t).split()) for t in df_cut['body'])
        unique_words = len(set(w for t in df_cut['body'] for w in str(t).split()))

        data_tos_save = data_tos_save._append({
            "Index_name": i,
            "Name": model_name,
            "f1_weighted": f1_weighted,
            "f1_micro": f1_macro,
            "f1_macro": f1_micro,
            "length": len(df_cut),
            "unique_label": len(df_cut["RGNTI3"].unique().tolist()),
            "string_length_average": np.mean(text_lengths),
            "string_length_median": np.median(text_lengths),
            'доля_спец_токенов': special_tokens / total_words if total_words > 0 else 0,
            'лекс_разнообразие': unique_words / total_words if total_words > 0 else 0

        }, ignore_index=True)
        data_tos_save.to_excel("Results_level_3_big.xlsx")
        print(
            f'Название темы: {model_name}, Код темы: {i}, F1_weighted: {f1_weighted:.2f}, F1_macro: {f1_macro:.2f}, '
            f'F1_micro: {f1_micro:.2f}, Длина датасета: {len(df_cut)}, '
            f'Уникальных классов: {len(df_cut["RGNTI3"].unique().tolist())}')
        joblib.dump(pipeline, f'/Users/denismazepa/Desktop/Py_projects/VKR/models/level_3_models/{i}_{model_name}.joblib')

print(f"Всего: {count}")
print(f"Среднее значение просто: {sums / count}")
print(f"Среднее значение без учета аномальных f1(по типу 1.00, 0.75 и т.д.): {sum_real / count_real}")
print(f"Несоотвествие таблиц кодов(нет имени): {count_no_name}")
print(f"Пропущено - маленькая выборка: {count_passed_small}(Выборка меньше 10)")
print(f"Пропущено - один класс: {count_passed_one_label} ")
print(f"Аномальное f1: {count_high_f1}")
print(f"Нуждается в аугментации: {count_need_augmentation}")

