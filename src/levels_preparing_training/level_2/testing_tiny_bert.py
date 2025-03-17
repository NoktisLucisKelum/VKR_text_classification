import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification

df = pd.read_csv(
    "/datasets/datasets_final/for_1_level/train_refactored_lematize_cut_final.csv",
    dtype={'RGNTI1': str, 'RGNTI2': str, 'RGNTI3': str})

df = df[df["RGNTI1"] == "44"]
print(len(df))
unique_labels = sorted(df['RGNTI2'].unique().tolist())
print(unique_labels)
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

df['label_id'] = df['RGNTI2'].apply(lambda x: label2id[x])

# ----------------------------------------------------------------------------
print("3. Разделение на train / val / test")
# ----------------------------------------------------------------------------
# train_df, test_df = train_test_split(
#     df, test_size=0.15, random_state=42, stratify=df['label_id']
# )
train_df, val_df = train_test_split(
    df, test_size=0.1, random_state=42, stratify=df['label_id']
)


# ----------------------------
# 3. Класс датасета (PyTorch)
# ----------------------------
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=1500):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Токенизация
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        # encoding['input_ids'] -> тензор размерности [1, max_len]
        # нам удобно вернуть "сплющенные" тензоры [max_len], поэтому возьмём .squeeze()
        item = {key: val.squeeze() for key, val in encoding.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)

        return item


# ----------------------------
print("4. Создаем датасеты и DataLoader-ы")
# ----------------------------

model_name = "cointegrated/rubert-tiny2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_dataset = TextDataset(
    texts=train_df['body'].tolist(),
    labels=train_df['label_id'].tolist(),
    tokenizer=tokenizer
)

val_dataset = TextDataset(
    texts=val_df['body'].tolist(),
    labels=val_df['label_id'].tolist(),
    tokenizer=tokenizer
)
# test_dataset = TextDataset(texts=test_df['body'].tolist(),
#                            labels=test_df['label_id'].tolist(),
#                            tokenizer=tokenizer
#                            )

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ----------------------------
print("5. Инициализация модели")
# ----------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(unique_labels)  # важно указать, сколько у нас классов
)

# Перенесём на GPU при возможности
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Определим оптимизатор и функцию потерь
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()


print("6. Цикл обучения")

epochs = 3  # Для примера

for epoch in range(epochs):
    print(f"\n=== Epoch {epoch + 1}/{epochs} ===")

    # ---- Тренировка ----
    model.train()
    running_loss = 0.0

    for step, batch in enumerate(train_loader):
        # Получаем входные данные
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        # Обратный проход
        loss.backward()
        optimizer.step()

        # Сохраняем статистику
        running_loss += loss.item()

        # ---- Вычислим F1 (weighted) на текущем batch ----
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        gold = labels.cpu().numpy()
        f1_weighted = f1_score(gold, preds, average='weighted')
        f1_macro = f1_score(gold, preds, average='macro')
        f1_micro = f1_score(gold, preds, average='micro')

        # Выведем статистику
        print(f"Batch {step + 1}/{len(train_loader)}, Loss: {loss.item():.4f}, F1 (weighted): {f1_weighted:.4f}, "
              f"F1 (macro): {f1_macro:.4f}, F1 (micro): {f1_micro:.4f}")

    # ---- Валидация на val_loader в конце эпохи (по желанию) ----
    model.eval()
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            gold = labels.cpu().numpy()

            val_preds.extend(preds)
            val_labels.extend(gold)

    val_f1_weighted = f1_score(val_labels, val_preds, average='weighted')
    val_f1_macro = f1_score(val_labels, val_preds, average='macro')
    val_f1_micro = f1_score(val_labels, val_preds, average='micro')
    print(f"Validation F1 (weighted): {val_f1_weighted:.4f}, Validation F1 (macro): {val_f1_macro:.4f},"
          f" Validation F1 (micro): {val_f1_micro:.4f}")

# ----------------------------
print("7. Сохранение модели")
# ----------------------------
# Сохраним модель и токенайзер в папку rubert_tiny2_model
save_path = "rubert_tiny2_model"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

# ----------------------------
print("8. Загрузка модели и пример предсказания")
# ----------------------------
loaded_tokenizer = AutoTokenizer.from_pretrained(save_path)
loaded_model = AutoModelForSequenceClassification.from_pretrained(save_path)
loaded_model.to(device)
loaded_model.eval()


def predict_class(text):
    """Функция предсказания класса на одном примере текста"""
    inputs = loaded_tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = loaded_model(**inputs)
        logits = outputs.logits
        # Применим softmax, чтобы получить вероятности
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_label_id = np.argmax(probs)
        pred_label = id2label[pred_label_id]
        confidence = probs[pred_label_id]
    return pred_label, confidence

    # Протестируем на некотором тексте


test_text = ("Влияние изменения параметров электрической сети с гибкими линиями электропередачи и асинхронными "
             "генераторами двойного питания на статическую устойчивость	Рассматривается эффективность ВЭС и СЭС в "
             "условиях истощения запасов традиционных видов топлива. В связи с значительным увеличением числа ВЭС "
             "важным является обеспечение устойчивости связанных с сетью асинхронных генераторов двойного питания, "
             "находящихся на валу ветротурбины.")
pred_label, conf = predict_class(test_text)
print(f"Текст: {test_text}")
print(f"Предсказанный класс: {pred_label}, вероятность: {conf:.4f}")
