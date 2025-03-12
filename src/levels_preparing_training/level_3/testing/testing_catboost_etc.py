from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, make_scorer
# from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from gensim.models import Word2Vec, FastText
import numpy as np
from sklearn.pipeline import Pipeline
import joblib
from random import randint
from sklearn.model_selection import GridSearchCV

# Загрузите ваш датасет
df = pd.read_csv(
    '/Users/denismazepa/Desktop/Py_projects/VKR/datasets/datasets_final/train_refactored_lematize_cut_final.csv',
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

# df_10 = df[(df['RGNTI1'] == '15') & (df['RGNTI2'] == '15.25')]


# df_1.to_csv("df_1.csv")
# df_2.to_csv("df_2.csv")
# df_3.to_csv("df_3.csv")
# df_4.to_csv("df_4.csv")
# df_5.to_csv("df_5.csv")
print(len(df_1), len(df_2), len(df_3), len(df_4), len(df_5), len(df_6), len(df_7), len(df_8), len(df_9), len(df_10),
      len(df_11), len(df_12), len(df_13), len(df_14), len(df_15))
# print(df_10.head(5))
#
#
# def catboost(df: pd.DataFrame):
#     # Разделите данные на признаки (тексты) и метки (классы)
#     X = df['body']
#     y = df['RGNTI3']
#     print(1)
#     # Разделите данные на обучающую и тестовую выборки
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     print(2)
#     # Преобразуем текстовые данные в TF-IDF признаки
#     vectorizer = TfidfVectorizer(max_features=10000)  # Ограничьте количество признаков для скорости
#     X_train_tfidf = vectorizer.fit_transform(X_train)
#     X_test_tfidf = vectorizer.transform(X_test)
#     print(3)
#     # Инициализируем и обучаем модель CatBoost
#     model = CatBoostClassifier(iterations=4000, learning_rate=0.1, depth=6)
#     model.fit(X_train_tfidf, y_train)
#     print(4)
#     # Делайте предсказания и оцените модель
#     y_pred = model.predict(X_test_tfidf)
#     print("Accuracy:", accuracy_score(y_test, y_pred))
#     f1 = f1_score(y_test, y_pred, average='weighted')
#     print(f'F1 Score: {f1:.2f}')
#
#     # Вычисление ROC AUC
#     print("Classification Report:n", classification_report(y_test, y_pred))


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

dict_of_models = {"RandomForestClassifier": RandomForestClassifier(),
                  "LinearSVC()": LinearSVC(),
                  "LogisticRegression": LogisticRegression(max_iter=2000),
                  "GradientBoostingClassifier": GradientBoostingClassifier(),
                  # "MultinomialNB": MultinomialNB() плохой результат
                  "ExtraTreesClassifier": ExtraTreesClassifier(),
                  # "AdaBoostClassifier": AdaBoostClassifier() плохой результат
                  "KNeighborsClassifier": KNeighborsClassifier(),
                  "DecisionTreeClassifier": DecisionTreeClassifier(),
                  # "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss') посредственный ре
                  }
vectorizers = {"TfidfVectorizer": TfidfVectorizer(max_features=5000), "CountVectorizer": CountVectorizer()}


def usual_models() -> None:
    average_f1, small_f1, big_f1 = 0, 0, 0
    best_variant_all, best_variant_small, best_variant_big = "", "", ""
    for k in dict_of_models.keys():
        for j in vectorizers.keys():
            sum_f1 = 0
            sum_f1_big = 0
            sum_f1_small = 0
            for i in dict_of_frames.keys():
                X = vectorizers[j].fit_transform(dict_of_frames[i][0]['body'])
                y = dict_of_frames[i][0]['RGNTI3']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = dict_of_models[k]
                model.fit(X_train, y_train)
                # Предсказание меток классов
                y_pred = model.predict(X_test)
                f1 = f1_score(y_test, y_pred, average='weighted')

                sum_f1 += f1
                if len(dict_of_frames[i][0]) > 1000:
                    sum_f1_big += f1
                else:
                    sum_f1_small += f1
                print(
                    f'Модель: {k}, Векторизатор: {j}, Датасет: {i}, F1 Score: {f1:.3f}, Длина датасета: {len(dict_of_frames[i][0])},'
                    f' Количество классов в датасете:  {len(dict_of_frames[i][0]['RGNTI3'].unique().tolist())}, Тема: {dict_of_frames[i][1]}')
            print("|||||||||||||||||||||||||||||||||||||")
            if sum_f1 > average_f1:
                average_f1 = sum_f1
                best_variant_all = f"Лучшая модель в общем: {k}, Лучший векторизатор: {j}, F1: средняя {average_f1 / 15}"
            if sum_f1_big > big_f1:
                big_f1 = sum_f1_big
                best_variant_big = f"Лучшая модель для больших датасетов: {k}, Лучший векторизатор: {j}, F1: средняя {big_f1 / 5}"
            if sum_f1_small > small_f1:
                small_f1 = sum_f1_small
                best_variant_small = f"Лучшая модель для маленьких датасетов: {k}, Лучший векторизатор: {j}, F1: средняя {small_f1 / 10}"
        print("_______________________________________")
    print(best_variant_all)
    print(best_variant_big)
    print(best_variant_small)


dict_of_models_boosted = {"RandomForestClassifier": RandomForestClassifier(),
                          "LinearSVC()": LinearSVC(),
                          "LogisticRegression": LogisticRegression(max_iter=2000),
                          "GradientBoostingClassifier": GradientBoostingClassifier(),
                          # "MultinomialNB": MultinomialNB() плохой результат
                          "ExtraTreesClassifier": ExtraTreesClassifier()
                          # "AdaBoostClassifier": AdaBoostClassifier() плохой результат
                          # "KNeighborsClassifier": KNeighborsClassifier(), плохой результат
                          # "DecisionTreeClassifier": DecisionTreeClassifier(), плохой результат
                          # "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss') посредственный результат
                          }

dict_of_boosted_vectorizers = {
    "Word2Vec_CBOW": Word2Vec(vector_size=200, window=10, min_count=1, workers=4, negative=4),
    "Word2Vec_SkipGramm": Word2Vec(vector_size=200, window=10, min_count=1, workers=4, negative=4, sg=1),
    "FastText_CBOW": FastText(vector_size=200, window=10, min_count=1, workers=4, negative=4),
    "FastText_SkipGram": FastText(vector_size=200, window=10, min_count=1, workers=4, negative=4, sg=1)}


def boosted_vectorizers() -> None:
    average_f1, small_f1, big_f1 = 0, 0, 0
    best_variant_all, best_variant_small, best_variant_big = "", "", ""
    for k in dict_of_models_boosted.keys():
        for j in dict_of_boosted_vectorizers.keys():
            sum_f1 = 0
            sum_f1_big = 0
            sum_f1_small = 0
            for i in dict_of_frames.keys():
                sentences = [text.split() for text in dict_of_frames[i][0]['body']]
                if j == "FastText_CBOW":
                    model = FastText(sentences, vector_size=200, window=10, min_count=1, workers=4, negative=4)
                elif j == "Word2Vec_CBOW":
                    model = Word2Vec(sentences, vector_size=250, window=12, min_count=1, workers=4, negative=4)
                elif j == "Word2Vec_SkipGramm":
                    model = Word2Vec(sentences, vector_size=200, window=10, min_count=1, workers=4, negative=4, sg=1)
                elif j == "FastText_SkipGram":
                    model = FastText(sentences, vector_size=200, window=10, min_count=1, workers=4, negative=4, sg=1)

                def document_vector(doc):
                    return np.mean([model.wv[word] for word in doc if word in model.wv], axis=0)

                X = np.array([document_vector(sentence) for sentence in sentences])
                y = dict_of_frames[i][0]['RGNTI3']

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                classifier = dict_of_models[k]
                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)

                f1 = f1_score(y_test, y_pred, average="weighted")

                sum_f1 += f1
                if len(dict_of_frames[i][0]) > 1000:
                    sum_f1_big += f1
                else:
                    sum_f1_small += f1
                print(
                    f'Модель: {k}, Датасет: {i}, Векторизатор: {j}, F1 Score: {f1:.3f}, Длина датасета: {len(dict_of_frames[i][0])},'
                    f'Количество классов в датасете:  {len(dict_of_frames[i][0]['RGNTI3'].unique().tolist())}, Тема: {dict_of_frames[i][1]}')
            print("_______________________________________")
            if sum_f1 > average_f1:
                average_f1 = sum_f1
                best_variant_all = f"Лучшая модель в общем: {k}, Лучший векторизатор: {j}, F1: средняя {average_f1 / 15}"
            if sum_f1_big > big_f1:
                big_f1 = sum_f1_big
                best_variant_big = f"Лучшая модель для больших датасетов: {k}, Лучший векторизатор: {j}, F1: средняя {big_f1 / 5}"
            if sum_f1_small > small_f1:
                small_f1 = sum_f1_small
                best_variant_small = f"Лучшая модель для маленьких датасетов: {k}, Лучший векторизатор: {j}, F1: средняя {small_f1 / 10}"
        print("|||||||||||||||||||||||||||||||||||||||")
    print(best_variant_all)
    print(best_variant_big)
    print(best_variant_small)


def gridsearch_word2vec() -> None:
    average_f1, small_f1, big_f1 = 0, 0, 0
    best_variant_all, best_variant_small, best_variant_big = "", "", ""
    for k in dict_of_models_boosted.keys():
        for neg in [1, 3, 6]:
            for vect in [150, 200, 250]:
                for wind in [10,  12, 15]:
                    sum_f1, sum_f1_big, sum_f1_small = 0, 0, 0
                    for i in dict_of_frames.keys():

                        sentences = [text.split() for text in dict_of_frames[i][0]['body']]
                        fst = Word2Vec(sentences, vector_size=vect, window=wind, min_count=1, workers=4, negative=neg,
                                       sg=1)

                        def document_vector(doc):
                            return np.mean([fst.wv[word] for word in doc if word in fst.wv], axis=0)

                        X = np.array([document_vector(sentence) for sentence in sentences])
                        y = dict_of_frames[i][0]['RGNTI3']

                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

                        classifier = dict_of_models[k]
                        classifier.fit(X_train, y_train)

                        y_pred = classifier.predict(X_test)

                        f1 = f1_score(y_test, y_pred, average="weighted")
                        sum_f1 += f1
                        if len(dict_of_frames[i][0]) > 1000:
                            sum_f1_big += f1
                        else:
                            sum_f1_small += f1
                        print(
                            f'Модель: {k}, Датасет: {i}, Векторизатор: Word2VecSG, F1 Score: {f1:.3f}, negative: {neg},'
                            f' window: {wind}, vectorsize: {vect}, Длина датасета: {len(dict_of_frames[i][0])},'
                            f'Количество классов в датасете:  {len(dict_of_frames[i][0]['RGNTI3'].unique().tolist())},'
                            f' Тема: {dict_of_frames[i][1]}')
                    if sum_f1 > average_f1:
                        average_f1 = sum_f1
                        best_variant_all = (f"Лучшие параметры для всех датасетов: negative: {neg} window: {wind},"
                                            f" vectorsize: {vect}, F1: средняя {average_f1 / 5}")
                    if sum_f1_big > big_f1:
                        big_f1 = sum_f1_big
                        best_variant_big = (f"Лучшие параметры для больших датасетов: negative: {neg} window: {wind},"
                                            f" vectorsize: {vect}, F1: средняя {big_f1 / 5}")
                    if sum_f1_small > small_f1:
                        small_f1 = sum_f1_small
                        best_variant_small = (f"Лучшие параметры для маленьких датасетов: negative: {neg} window: {wind},"
                                              f"vectorsize: {vect},F1: средняя {small_f1 / 10}")
                    print("_______________________________________")
                    print()
    print(best_variant_all)
    print(best_variant_big)
    print(best_variant_small)


def save_search():
    for i in [df_13, df_7]:
        X_train, X_test, y_train, y_test = train_test_split(i['body'], i['RGNTI3'], test_size=0.2, random_state=42)
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('svc', LinearSVC())
        ])
        pipeline.fit(X_train, y_train)
        print(f'Accuracy: {pipeline.score(X_test, y_test) * 100:.2f}%')
        joblib.dump(pipeline, f'text_classification_{randint(1, 5)}.joblib')


grid_search_list = [df_1, df_7, df_14]


def grid_search_scv():
    for i in range(0, 3):
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('svc', LinearSVC())
        ])

        parameters = {'tfidf__max_df': [0.5, 0.65, 0.7],
                      'tfidf__ngram_range': [(1, 1), (1, 2)],
                      'tfidf__use_idf': [True, False],
                      'svc__C': [0.65, 0.75, 0.85],
                      'svc__penalty': ['l2'],  # Тип регуляризации (l1 не поддерживается без dual=False)
                      'svc__loss': ['squared_hinge'],  # Функция потерь
                      'svc__fit_intercept': [True, False],  # Добавлять ли смещение?
                      'svc__class_weight': ['balanced'],  # Веса классов
                      'svc__tol': [1e-4],
                      }

        grid_search = GridSearchCV(pipeline, parameters, scoring='f1_macro', cv=5, n_jobs=-1, verbose=1)

        X = grid_search_list[i]['body']
        y = grid_search_list[i]['RGNTI3']
        print(X.head(5), y.head(5))

        grid_search.fit(X, y)
        print(f"Номер датасета: {i}")
        print("Наилучшие параметры:")
        print(grid_search.best_params_)

        print("Наилучшее значение метрики f1:")
        print(grid_search.best_score_)


def testing_after_gridsearch():
    for i in dict_of_frames.keys():
        X_train, X_test, y_train, y_test = train_test_split(dict_of_frames[i][0]['body'],
                                                            dict_of_frames[i][0]['RGNTI3'],
                                                            test_size=0.2, random_state=42)
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_df=0.5, ngram_range=(1, 1), use_idf=True)),
            ('svc', LinearSVC(C=0.65, class_weight='balanced', fit_intercept=False, loss='squared_hinge', max_iter=4000,
                              penalty='l2', tol=0.0001))
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="weighted")
        print(
            f' Датасет: {i}, Векторизатор: Word2VecSG, F1 Score: {f1:.3f}, Длина датасета: {len(dict_of_frames[i][0])},'
            f'Количество классов в датасете:  {len(dict_of_frames[i][0]['RGNTI3'].unique().tolist())},'
            f' Тема: {dict_of_frames[i][1]}')
        joblib.dump(pipeline, f'text_classification_{dict_of_frames[i][1]}.joblib')

# usual_models()
# boosted_vectorizers()
# gridsearch_word2vec()
# save_search()

# grid_search_scv()

# testing_after_gridsearch()


def testing_tfidf_upgrade():
    for i in dict_of_frames.keys():
        X_train, X_test, y_train, y_test = train_test_split(dict_of_frames[i][0]['body'],
                                                            dict_of_frames[i][0]['RGNTI3'],
                                                            test_size=0.2, random_state=42)
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_df=0.5, ngram_range=(1, 1), use_idf=True)),
            ('svc', LinearSVC(C=0.65, class_weight='balanced', fit_intercept=False, loss='squared_hinge', max_iter=4000,
                              penalty='l2', tol=0.0001))
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="weighted")
        print(
            f' Датасет: {i}, Векторизатор: TF-IDF, F1 Score: {f1:.3f}, Длина датасета: {len(dict_of_frames[i][0])},'
            f'Количество классов в датасете:  {len(dict_of_frames[i][0]['RGNTI3'].unique().tolist())},'
            f'  Тема: {dict_of_frames[i][1]}')


testing_tfidf_upgrade()