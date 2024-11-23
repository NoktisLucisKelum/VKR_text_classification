import pandas as pd
from sklearn.pipeline import Pipeline
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import json
df = pd.read_csv(
    "/Users/denismazepa/Desktop/Py_projects/VKR/datasets/datasets_final/train_refactored_lematize_cut_final.csv",
    dtype={'RGNTI1': str, 'RGNTI2': str, 'RGNTI3': str})

with open("/Users/denismazepa/Desktop/Py_projects/VKR/grnti/GRNTI_2_ru.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

list_of_unique = df["RGNTI2"].unique().tolist()
print(list_of_unique)
for i in list_of_unique:
    model_name = data[i].replace(' ', '_')
    df_cut = df[df["RGNTI2"] == i]
    if len(df_cut["RGNTI3"].unique().tolist()) != 1:
        X_train, X_test, y_train, y_test = train_test_split(df_cut['body'], df_cut['RGNTI3'], test_size=0.2, random_state=42)
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('svc', LinearSVC())
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="weighted")
        print(f'F1: {f1:.2f}%')
        joblib.dump(pipeline, f'/Users/denismazepa/Desktop/Py_projects/VKR/models/level_3_models/{model_name}.joblib')

        print(model_name)



