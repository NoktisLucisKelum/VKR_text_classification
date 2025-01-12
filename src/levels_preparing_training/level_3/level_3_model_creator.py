import pandas as pd
from sklearn.pipeline import Pipeline
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

warnings.filterwarnings("ignore")

df = pd.read_csv(
    "/Users/denismazepa/Desktop/Py_projects/VKR/datasets/datasets_final/train_refactored_lematize_cut_final.csv",
    dtype={'RGNTI1': str, 'RGNTI2': str, 'RGNTI3': str})
print(0)

with open("/Users/denismazepa/Desktop/Py_projects/VKR/grnti/GRNTI_2_ru.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

print(1)
df_reserve = pd.read_csv(
    "/Users/denismazepa/Desktop/Py_projects/VKR/datasets/datatsets_from_git/train/train_ru_work.csv",
    sep="\t", on_bad_lines='skip')

MODEL_NAME = 'cointegrated/rut5-base-paraphraser'
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(2)

pipeline_0 = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.5, ngram_range=(1, 1), use_idf=True)),
    ('svc', LinearSVC(C=0.65, class_weight='balanced', fit_intercept=False, loss='squared_hinge', max_iter=4000,
                      penalty='l2', tol=0.0001))
])


def augment_dataset_with_t5(dataset: pd.DataFrame, text_column: str, class_column: str, iter: int) -> pd.DataFrame:
    """
    Увеличивает размер датасета, если его длина меньше 1000, добавляя новые строки с перефразированными текстами.

    :param dataset: pandas DataFrame с текстами и классами
    :param text_column: Название столбца с текстами
    :param class_column: Название столбца с классами
    :return: Обновленный DataFrame
    """

    dataset_length, unique_classes, class_counts = (len(dataset), dataset[class_column].nunique(),
                                                    dataset[class_column].value_counts())

    # print(f"Длина датасета: {dataset_length}")
    # print(f"Количество уникальных классов: {unique_classes}")
    # print(f"Количество значений для каждого класса:\n{class_counts}")
    print("Увеличиваем размер датасета...")
    augmented_data = []

    for cls in class_counts.index:
        class_texts = dataset[dataset[class_column] == cls][text_column]

        for text in class_texts:
            for _ in range(iter):
                try:
                    new_t = time()
                    new_text = paraphrase(text)
                    end_t = time()
                    print(f"Elapsed_time: {end_t - new_t}")
                    augmented_data.append({text_column: new_text, class_column: cls})
                except Exception as e:
                    print(f"Ошибка при работе с ruGPT-3: {e}")
                    continue

    augmented_df = pd.DataFrame(augmented_data)
    updated_dataset = pd.concat([dataset, augmented_df], ignore_index=True)

    print(f"Новый размер датасета: {len(updated_dataset)}")
    return updated_dataset


def paraphrase(text, beams=5, grams=4):
    x = tokenizer(text, return_tensors='pt', padding=True).to(model.device)
    max_size = int(x.input_ids.shape[1] * 1.5 + 10)
    out = model.generate(**x, encoder_no_repeat_ngram_size=grams, num_beams=beams, max_length=max_size)
    return tokenizer.decode(out[0], skip_special_tokens=True)


list_of_unique = df["RGNTI2"].unique().tolist()
dict_of_success = {}
sum, count, count_high_f1 = 0, 0, 0
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
        if f1_weighted < 0.73 or f1_micro == 1.00 or f1_macro == 1.00 or f1_weighted == 1.00:
            count_need_augmentation += 1
            cycle_len = 1
            print("___________________________________________________________")
            print(f"Недостаточный f1: {f1_weighted:.2f}")
            new_df = df_reserve[df_reserve["RGNTI2"].str.contains(i)]

            preprocessor_iter_1 = TextPreprocessor(new_df)
            preprocessor_iter_1.drop_nan()
            preprocessor_iter_1.merge_and_drop_columns('body', 'title')

            iter_1 = preprocessor_iter_1.return_dataset()
            iter_2 = augment_dataset_with_t5(iter_1, 'body', 'RGNTI3', cycle_len)
            iter_2.dropna()

            preprocessor_iter_2 = TextPreprocessor(iter_2)
            preprocessor_iter_2.chem_formula_prepare()
            preprocessor_iter_2.phys_formula_prepare()
            preprocessor_iter_2.remove_english_strings()
            preprocessor_iter_2.lemmatize(['body', 'keywords'])
            preprocessor_iter_2.remove_punctuation(['body'])
            # # preprocessor.stem(['text_column'])
            preprocessor_iter_2.remove_stop_words(['body'])
            # preprocessor.convert_to_word_list(['title', 'body', 'keywords'])
            preprocessor_iter_2.remove_second_index(["RGNTI1", "RGNTI2", "RGNTI3"])
            preprocessor_iter_2.drop_columns(["correct"])
            # preprocessor_iter_2.printing("keywords")
            # preprocessor_iter_2.printing("body")
            preprocessor_iter_2.repare_columns()
            iter_3 = preprocessor_iter_2.return_dataset()
            print(f"Длина старого датасета: {len(df_cut)}, Длинна датасета который достали из "
                  f"изнчальной базы: {len(new_df)}, Длинна нового датасета: {len(iter_3)}")
            print(f'Название темы: {model_name}, F1_новый: {f1_weighted:.2f}, Длина датасета: {len(iter_3)}, '
                  f"Уникальных классов: {len(iter_3['RGNTI3'].unique().tolist())} Точность недостаточная")
            X_train, X_test, y_train, y_test = train_test_split(iter_3['body'], iter_3['RGNTI3'],
                                                                test_size=0.2, random_state=42)
            pipeline_new = pipeline
            pipeline_new.fit(X_train, y_train)
            y_pred = pipeline_new.predict(X_test)
            f1_weighted_new = f1_score(y_test, y_pred, average="weighted")
            f1_macro_new = f1_score(y_test, y_pred, average="macro")
            f1_micro_new = f1_score(y_test, y_pred, average="micro")

            sum += f1_weighted_new
            count += 1
            dict_of_success[i] = f1_weighted_new

            print(f'Обновленный датасет. Название темы: {model_name}, F1_weighted: {f1_weighted_new:.2f}%, '
                  f'F1_macro: {f1_macro_new:.2f}, F1_micro: {f1_micro_new:.2f}, '
                  f'Длина датасета: {len(iter_3)},Уникальных классов: {len(iter_3["RGNTI3"].unique().tolist())}')
            joblib.dump(pipeline_new,
                        f'/Users/denismazepa/Desktop/Py_projects/VKR/models/level_3_models/{model_name}.joblib')
            print("___________________________________________________________")
        else:
            sum += f1_weighted
            count += 1
            dict_of_success[i] = f1_weighted
            print(
                f'Название темы: {model_name}, Код темы: {i}, F1_weighted: {f1_weighted:.2f}, F1_macro: {f1_macro:.2f}, '
                f'F1_micro: {f1_micro:.2f}, Длина датасета: {len(df_cut)}, '
                f'Уникальных классов: {len(df_cut["RGNTI3"].unique().tolist())}')
        joblib.dump(pipeline, f'/Users/denismazepa/Desktop/Py_projects/VKR/models/level_3_models/{model_name}.joblib')

print(f"Всего: {count}")
print(f"Среднее значение: {sum / count}")
print(f"Среднее значение без учета аномальных f1(по типу 1.00, 0.75 и т.д.): {sum_real / count_real}")
print(f"Несоотвествие таблиц кодов(нет имени): {count_no_name}")
print(f"Пропущено - маленькая выборка: {count_passed_small} ")
print(f"Пропущено - один класс: {count_passed_one_label} ")
print(f"Аномальное f1: {count_high_f1}")
print(f"Нуждается в аугментации: {count_need_augmentation}")
