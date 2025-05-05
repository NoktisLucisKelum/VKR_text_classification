import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, f1_score
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from sklearn.svm import LinearSVC

# 1. Загрузка данных
df = pd.read_csv(
    "/Users/denismazepa/Desktop/Py_projects/VKR/datasets/datasets_final/for_other_levels/train_refactored_lematize_no_numbers_2_3_level.csv",
    dtype={'RGNTI1': str, 'RGNTI2': str, 'RGNTI3': str})

df_1 = df[(df['RGNTI1'] == '55') & (df['RGNTI2'] == '55.39')]
texts = df_1['body'].tolist()
labels = df_1['RGNTI3'].tolist()
print(len(df_1))
# 2. Разделение на train/test (стратифицировано по меткам)
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)
print(1)
# 3. Загрузка предобученного эмбеддера (подберите под ваш язык)
device = 'cpu'

dict_of_sentence_transformers = {"all-mpnet-base-v2": SentenceTransformer("all-mpnet-base-v2", device="cpu"),
                                 "all-MiniLM-L12-v2": SentenceTransformer("all-MiniLM-L12-v2", device="cpu"),
                                 "paraphrase-multilingual-mpnet-base-v2": SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device="cpu"),
                                 "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device='cpu')}


print(2)


# 4. Преобразование текстов в эмбеддинги (батчами для экономии памяти)
def encode_texts(texts, model, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        batch_emb = model.encode(batch, show_progress_bar=False)
        embeddings.append(batch_emb)
    return np.vstack(embeddings)


for i in dict_of_sentence_transformers.keys():
    print("Создание эмбеддингов...")
    model = dict_of_sentence_transformers[i]
    X_train_emb = encode_texts(X_train, model)
    X_test_emb = encode_texts(X_test, model)
    print(3)
    # 5. Обучение классификатора (выберите один)
    # Вариант 1: LogisticRegression (быстрее, лучше для многих классов)
    clf = LinearSVC(
        C=0.2,
        class_weight='balanced',
        max_iter=5000,
        dual=False,
        verbose=1,
        random_state=42
    )

    # Вариант 2: KNeighborsClassifier (медленнее, но интерпретируемо)
    # clf = KNeighborsClassifier(n_neighbors=5, metric='cosine')
    print(4)
    clf.fit(X_train_emb, y_train)
    print(5)
    # 6. Предсказание и оценка
    y_pred = clf.predict(X_test_emb)

    # Отчёт по классификации (F1-macro ключевая)
    # print(classification_report(y_test, y_pred, zero_division=0))
    print(f"Модель: {i}, F1-macro: {f1_score(y_test, y_pred, average='weighted'):.4f}")

    # 7. Пример предсказания для нового текста
    new_text = "Пример текста для классификации"
    new_embedding = model.encode([new_text])
    predicted_label = clf.predict(new_embedding)[0]
    print(f"Предсказанный класс: {predicted_label}")