import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
from equal_df import select_100_per_group

from transformers import AutoTokenizer, AutoModelForSequenceClassification

df = pd.read_csv(
    "/Users/denismazepa/Desktop/Py_projects/VKR/datasets/datasets_final/for_1_level/train_refactored_lematize_no_numbers_1_level_cut.csv",
    dtype={'RGNTI1': str, 'RGNTI2': str, 'RGNTI3': str})

df = select_100_per_group(df, "RGNTI1")
unique_labels = sorted(df['RGNTI1'].unique().tolist())
print(unique_labels)
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

df['label_id'] = df['RGNTI1'].apply(lambda x: label2id[x])

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
    def __init__(self, texts, labels, tokenizer, max_len=120):
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

model_name = "DeepPavlov/rubert-base-cased-sentence"
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
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = nn.CrossEntropyLoss()
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

print("6. Цикл обучения")

epochs = 2  # Для примера

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
        # scheduler.step()

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
save_path = "rubert_sentence_model"
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
"""с использованием lr"""
# Batch 1/805, Loss: 3.7221, F1 (weighted): 0.0000, F1 (macro): 0.0000, F1 (micro): 0.0000
# Batch 2/805, Loss: 3.7219, F1 (weighted): 0.0074, F1 (macro): 0.0084, F1 (micro): 0.0625
# Batch 3/805, Loss: 3.8839, F1 (weighted): 0.0000, F1 (macro): 0.0000, F1 (micro): 0.0000
# Batch 4/805, Loss: 3.8294, F1 (weighted): 0.0000, F1 (macro): 0.0000, F1 (micro): 0.0000
# Batch 5/805, Loss: 3.7662, F1 (weighted): 0.0000, F1 (macro): 0.0000, F1 (micro): 0.0000
# Batch 6/805, Loss: 3.7858, F1 (weighted): 0.0000, F1 (macro): 0.0000, F1 (micro): 0.0000
# Batch 7/805, Loss: 3.7847, F1 (weighted): 0.0250, F1 (macro): 0.0333, F1 (micro): 0.0625
# Batch 8/805, Loss: 3.6409, F1 (weighted): 0.0000, F1 (macro): 0.0000, F1 (micro): 0.0000
# Batch 9/805, Loss: 3.7388, F1 (weighted): 0.0000, F1 (macro): 0.0000, F1 (micro): 0.0000
# Batch 10/805, Loss: 3.7491, F1 (weighted): 0.0000, F1 (macro): 0.0000, F1 (micro): 0.0000
# Batch 11/805, Loss: 3.7819, F1 (weighted): 0.0125, F1 (macro): 0.0133, F1 (micro): 0.0625
# Batch 12/805, Loss: 3.7792, F1 (weighted): 0.0156, F1 (macro): 0.0147, F1 (micro): 0.0625
# Batch 13/805, Loss: 3.8801, F1 (weighted): 0.0000, F1 (macro): 0.0000, F1 (micro): 0.0000
# Batch 14/805, Loss: 3.7194, F1 (weighted): 0.0089, F1 (macro): 0.0102, F1 (micro): 0.0625
# Batch 15/805, Loss: 3.8300, F1 (weighted): 0.0000, F1 (macro): 0.0000, F1 (micro): 0.0000
# Batch 16/805, Loss: 3.7903, F1 (weighted): 0.0000, F1 (macro): 0.0000, F1 (micro): 0.0000
# Batch 17/805, Loss: 3.7062, F1 (weighted): 0.0000, F1 (macro): 0.0000, F1 (micro): 0.0000
# Batch 18/805, Loss: 3.8077, F1 (weighted): 0.0074, F1 (macro): 0.0098, F1 (micro): 0.0625
# Batch 19/805, Loss: 3.7878, F1 (weighted): 0.0000, F1 (macro): 0.0000, F1 (micro): 0.0000
# Batch 20/805, Loss: 3.7799, F1 (weighted): 0.0000, F1 (macro): 0.0000, F1 (micro): 0.0000
# Batch 21/805, Loss: 3.6849, F1 (weighted): 0.0000, F1 (macro): 0.0000, F1 (micro): 0.0000
# Batch 22/805, Loss: 3.7697, F1 (weighted): 0.0000, F1 (macro): 0.0000, F1 (micro): 0.0000
# Batch 23/805, Loss: 3.8288, F1 (weighted): 0.0000, F1 (macro): 0.0000, F1 (micro): 0.0000
# Batch 24/805, Loss: 3.6056, F1 (weighted): 0.0104, F1 (macro): 0.0104, F1 (micro): 0.0625
# Batch 25/805, Loss: 3.6860, F1 (weighted): 0.0000, F1 (macro): 0.0000, F1 (micro): 0.0000
# Batch 26/805, Loss: 3.8535, F1 (weighted): 0.0000, F1 (macro): 0.0000, F1 (micro): 0.0000
# Batch 27/805, Loss: 3.7817, F1 (weighted): 0.0801, F1 (macro): 0.0812, F1 (micro): 0.1875
# Batch 28/805, Loss: 3.7908, F1 (weighted): 0.0000, F1 (macro): 0.0000, F1 (micro): 0.0000
# Batch 29/805, Loss: 3.6900, F1 (weighted): 0.0461, F1 (macro): 0.0323, F1 (micro): 0.1250
# Batch 30/805, Loss: 3.7720, F1 (weighted): 0.0000, F1 (macro): 0.0000, F1 (micro): 0.0000
# Batch 31/805, Loss: 3.7702, F1 (weighted): 0.0000, F1 (macro): 0.0000, F1 (micro): 0.0000
# Batch 32/805, Loss: 3.5608, F1 (weighted): 0.0500, F1 (macro): 0.0222, F1 (micro): 0.1250
# Batch 33/805, Loss: 3.6848, F1 (weighted): 0.0000, F1 (macro): 0.0000, F1 (micro): 0.0000
# Batch 34/805, Loss: 3.6799, F1 (weighted): 0.0250, F1 (macro): 0.0267, F1 (micro): 0.0625
# Batch 35/805, Loss: 3.8682, F1 (weighted): 0.0000, F1 (macro): 0.0000, F1 (micro): 0.0000
# Batch 36/805, Loss: 3.8521, F1 (weighted): 0.0000, F1 (macro): 0.0000, F1 (micro): 0.0000
# Batch 37/805, Loss: 3.8975, F1 (weighted): 0.0000, F1 (macro): 0.0000, F1 (micro): 0.0000
# Batch 38/805, Loss: 3.6650, F1 (weighted): 0.0096, F1 (macro): 0.0096, F1 (micro): 0.0625
# Batch 39/805, Loss: 3.7576, F1 (weighted): 0.0000, F1 (macro): 0.0000, F1 (micro): 0.0000
# Batch 40/805, Loss: 3.7093, F1 (weighted): 0.0074, F1 (macro): 0.0078, F1 (micro): 0.0625
# Batch 41/805, Loss: 3.7766, F1 (weighted): 0.0000, F1 (macro): 0.0000, F1 (micro): 0.0000
# Batch 42/805, Loss: 3.7111, F1 (weighted): 0.0625, F1 (macro): 0.0333, F1 (micro): 0.1250
# Batch 43/805, Loss: 3.7982, F1 (weighted): 0.0000, F1 (macro): 0.0000, F1 (micro): 0.0000
# Batch 44/805, Loss: 3.7353, F1 (weighted): 0.0000, F1 (macro): 0.0000, F1 (micro): 0.0000
# Batch 45/805, Loss: 3.6098, F1 (weighted): 0.0208, F1 (macro): 0.0119, F1 (micro): 0.0625
# Batch 46/805, Loss: 3.6887, F1 (weighted): 0.0250, F1 (macro): 0.0250, F1 (micro): 0.0625
# Batch 47/805, Loss: 3.7141, F1 (weighted): 0.0156, F1 (macro): 0.0167, F1 (micro): 0.0625
# Batch 48/805, Loss: 3.7341, F1 (weighted): 0.0125, F1 (macro): 0.0133, F1 (micro): 0.0625
# Batch 49/805, Loss: 3.7270, F1 (weighted): 0.0208, F1 (macro): 0.0196, F1 (micro): 0.0625
# Batch 50/805, Loss: 3.7753, F1 (weighted): 0.0208, F1 (macro): 0.0185, F1 (micro): 0.0625
# Batch 51/805, Loss: 3.8578, F1 (weighted): 0.0000, F1 (macro): 0.0000, F1 (micro): 0.0000
# Batch 52/805, Loss: 3.8311, F1 (weighted): 0.0312, F1 (macro): 0.0278, F1 (micro): 0.0625
# Batch 53/805, Loss: 3.6229, F1 (weighted): 0.0125, F1 (macro): 0.0133, F1 (micro): 0.0625
# Batch 54/805, Loss: 3.6365, F1 (weighted): 0.0000, F1 (macro): 0.0000, F1 (micro): 0.0000
# Batch 55/805, Loss: 3.8437, F1 (weighted): 0.0000, F1 (macro): 0.0000, F1 (micro): 0.0000
# Batch 56/805, Loss: 3.6916, F1 (weighted): 0.0156, F1 (macro): 0.0167, F1 (micro): 0.0625
# Batch 57/805, Loss: 3.6886, F1 (weighted): 0.0417, F1 (macro): 0.0476, F1 (micro): 0.0625
# Batch 58/805, Loss: 3.8212, F1 (weighted): 0.0000, F1 (macro): 0.0000, F1 (micro): 0.0000
# Batch 59/805, Loss: 3.6266, F1 (weighted): 0.0000, F1 (macro): 0.0000, F1 (micro): 0.0000
# Batch 60/805, Loss: 3.7028, F1 (weighted): 0.0000, F1 (macro): 0.0000, F1 (micro): 0.0000
# Batch 61/805, Loss: 3.6810, F1 (weighted): 0.0625, F1 (macro): 0.0500, F1 (micro): 0.0625
# Batch 62/805, Loss: 3.6146, F1 (weighted): 0.1083, F1 (macro): 0.0881, F1 (micro): 0.1875
# Batch 63/805, Loss: 3.6707, F1 (weighted): 0.0000, F1 (macro): 0.0000, F1 (micro): 0.0000
# Batch 64/805, Loss: 3.6807, F1 (weighted): 0.0625, F1 (macro): 0.0625, F1 (micro): 0.0625
# Batch 65/805, Loss: 3.5049, F1 (weighted): 0.0875, F1 (macro): 0.0824, F1 (micro): 0.1250
# Batch 66/805, Loss: 3.6715, F1 (weighted): 0.0000, F1 (macro): 0.0000, F1 (micro): 0.0000
# Batch 67/805, Loss: 3.5357, F1 (weighted): 0.0982, F1 (macro): 0.0756, F1 (micro): 0.1250
# Batch 68/805, Loss: 3.6236, F1 (weighted): 0.0104, F1 (macro): 0.0111, F1 (micro): 0.0625
# Batch 69/805, Loss: 3.6121, F1 (weighted): 0.0000, F1 (macro): 0.0000, F1 (micro): 0.0000
# Batch 70/805, Loss: 3.5263, F1 (weighted): 0.0000, F1 (macro): 0.0000, F1 (micro): 0.0000
# Batch 71/805, Loss: 3.7195, F1 (weighted): 0.0000, F1 (macro): 0.0000, F1 (micro): 0.0000
# Batch 72/805, Loss: 3.5868, F1 (weighted): 0.0000, F1 (macro): 0.0000, F1 (micro): 0.0000
# Batch 73/805, Loss: 3.4830, F1 (weighted): 0.0670, F1 (macro): 0.0824, F1 (micro): 0.1875
# Batch 74/805, Loss: 3.2501, F1 (weighted): 0.3673, F1 (macro): 0.3092, F1 (micro): 0.4375
# Batch 75/805, Loss: 3.7228, F1 (weighted): 0.0000, F1 (macro): 0.0000, F1 (micro): 0.0000
# Batch 76/805, Loss: 3.4495, F1 (weighted): 0.0875, F1 (macro): 0.0824, F1 (micro): 0.1250
# Batch 77/805, Loss: 3.3700, F1 (weighted): 0.1187, F1 (macro): 0.0563, F1 (micro): 0.1250
# Batch 78/805, Loss: 3.5586, F1 (weighted): 0.0125, F1 (macro): 0.0100, F1 (micro): 0.0625
# Batch 79/805, Loss: 3.6800, F1 (weighted): 0.1250, F1 (macro): 0.1071, F1 (micro): 0.1250
# Batch 80/805, Loss: 3.3756, F1 (weighted): 0.0625, F1 (macro): 0.0588, F1 (micro): 0.0625
# Batch 81/805, Loss: 3.5561, F1 (weighted): 0.0000, F1 (macro): 0.0000, F1 (micro): 0.0000
# Batch 82/805, Loss: 3.3160, F1 (weighted): 0.0625, F1 (macro): 0.0556, F1 (micro): 0.1250
# Batch 83/805, Loss: 3.3596, F1 (weighted): 0.0250, F1 (macro): 0.0211, F1 (micro): 0.0625
# Batch 84/805, Loss: 3.4143, F1 (weighted): 0.0563, F1 (macro): 0.0450, F1 (micro): 0.1250
# Batch 85/805, Loss: 3.4195, F1 (weighted): 0.1500, F1 (macro): 0.0500, F1 (micro): 0.1250
# Batch 86/805, Loss: 3.3711, F1 (weighted): 0.0833, F1 (macro): 0.0370, F1 (micro): 0.0625
# Batch 87/805, Loss: 3.5415, F1 (weighted): 0.0312, F1 (macro): 0.0278, F1 (micro): 0.0625
# Batch 88/805, Loss: 3.3723, F1 (weighted): 0.2271, F1 (macro): 0.1426, F1 (micro): 0.2500
# Batch 89/805, Loss: 3.4008, F1 (weighted): 0.1042, F1 (macro): 0.0833, F1 (micro): 0.1250
# Batch 90/805, Loss: 3.3196, F1 (weighted): 0.1562, F1 (macro): 0.1078, F1 (micro): 0.1875
# Batch 91/805, Loss: 3.4242, F1 (weighted): 0.0500, F1 (macro): 0.0250, F1 (micro): 0.0625
# Batch 92/805, Loss: 3.2754, F1 (weighted): 0.1083, F1 (macro): 0.0912, F1 (micro): 0.1875
# Batch 93/805, Loss: 3.2615, F1 (weighted): 0.1250, F1 (macro): 0.1429, F1 (micro): 0.1875
# Batch 94/805, Loss: 3.2301, F1 (weighted): 0.1458, F1 (macro): 0.1167, F1 (micro): 0.1875
# Batch 95/805, Loss: 3.3088, F1 (weighted): 0.1875, F1 (macro): 0.1167, F1 (micro): 0.1875
# Batch 96/805, Loss: 3.2543, F1 (weighted): 0.1250, F1 (macro): 0.1000, F1 (micro): 0.1875
# Batch 97/805, Loss: 3.2742, F1 (weighted): 0.2292, F1 (macro): 0.1579, F1 (micro): 0.2500
# Batch 98/805, Loss: 3.2955, F1 (weighted): 0.0625, F1 (macro): 0.0556, F1 (micro): 0.0625
# Batch 99/805, Loss: 3.1230, F1 (weighted): 0.1667, F1 (macro): 0.1569, F1 (micro): 0.2500
# Batch 100/805, Loss: 3.1937, F1 (weighted): 0.2031, F1 (macro): 0.0921, F1 (micro): 0.2500
# Batch 101/805, Loss: 3.4819, F1 (weighted): 0.0000, F1 (macro): 0.0000, F1 (micro): 0.0000
# Batch 102/805, Loss: 3.2336, F1 (weighted): 0.1042, F1 (macro): 0.0614, F1 (micro): 0.1250
# Batch 103/805, Loss: 3.3216, F1 (weighted): 0.0875, F1 (macro): 0.0875, F1 (micro): 0.1250
# Batch 104/805, Loss: 3.3285, F1 (weighted): 0.1187, F1 (macro): 0.1056, F1 (micro): 0.1875
# Batch 105/805, Loss: 3.2495, F1 (weighted): 0.1354, F1 (macro): 0.1083, F1 (micro): 0.1875
# Batch 106/805, Loss: 3.3574, F1 (weighted): 0.0000, F1 (macro): 0.0000, F1 (micro): 0.0000
# Batch 107/805, Loss: 3.3093, F1 (weighted): 0.1771, F1 (macro): 0.0686, F1 (micro): 0.1250
# Batch 108/805, Loss: 3.0804, F1 (weighted): 0.2667, F1 (macro): 0.1647, F1 (micro): 0.3125
# Batch 109/805, Loss: 3.0664, F1 (weighted): 0.3583, F1 (macro): 0.2815, F1 (micro): 0.3750
# Batch 110/805, Loss: 3.2140, F1 (weighted): 0.2542, F1 (macro): 0.2140, F1 (micro): 0.3125
# Batch 111/805, Loss: 3.0922, F1 (weighted): 0.1667, F1 (macro): 0.1212, F1 (micro): 0.1875
# Batch 112/805, Loss: 2.7866, F1 (weighted): 0.2604, F1 (macro): 0.1889, F1 (micro): 0.3125
# Batch 113/805, Loss: 2.9851, F1 (weighted): 0.1562, F1 (macro): 0.1190, F1 (micro): 0.1875
# Batch 114/805, Loss: 3.0911, F1 (weighted): 0.1708, F1 (macro): 0.1088, F1 (micro): 0.1875
# Batch 115/805, Loss: 3.1719, F1 (weighted): 0.1042, F1 (macro): 0.0686, F1 (micro): 0.1250
# Batch 116/805, Loss: 3.3311, F1 (weighted): 0.0625, F1 (macro): 0.0476, F1 (micro): 0.0625
# Batch 117/805, Loss: 2.8781, F1 (weighted): 0.3333, F1 (macro): 0.2593, F1 (micro): 0.3125
# Batch 118/805, Loss: 3.0216, F1 (weighted): 0.0625, F1 (macro): 0.0526, F1 (micro): 0.0625
# Batch 119/805, Loss: 3.1061, F1 (weighted): 0.2500, F1 (macro): 0.1905, F1 (micro): 0.2500
# Batch 120/805, Loss: 3.1545, F1 (weighted): 0.2708, F1 (macro): 0.1404, F1 (micro): 0.2500
# Batch 121/805, Loss: 3.1275, F1 (weighted): 0.1667, F1 (macro): 0.1333, F1 (micro): 0.1875
# Batch 122/805, Loss: 2.8448, F1 (weighted): 0.1667, F1 (macro): 0.1000, F1 (micro): 0.1875
# Batch 123/805, Loss: 2.8857, F1 (weighted): 0.2917, F1 (macro): 0.2157, F1 (micro): 0.3125
# Batch 124/805, Loss: 3.0365, F1 (weighted): 0.1625, F1 (macro): 0.0929, F1 (micro): 0.1875
# Batch 125/805, Loss: 2.9410, F1 (weighted): 0.4313, F1 (macro): 0.2357, F1 (micro): 0.4375
# Batch 126/805, Loss: 3.0405, F1 (weighted): 0.2333, F1 (macro): 0.1604, F1 (micro): 0.2500
# Batch 127/805, Loss: 3.0055, F1 (weighted): 0.2589, F1 (macro): 0.1286, F1 (micro): 0.3125
# Batch 128/805, Loss: 3.0925, F1 (weighted): 0.1875, F1 (macro): 0.1429, F1 (micro): 0.1875
# Batch 129/805, Loss: 2.8400, F1 (weighted): 0.3229, F1 (macro): 0.1574, F1 (micro): 0.3125
# Batch 130/805, Loss: 2.5733, F1 (weighted): 0.6875, F1 (macro): 0.5238, F1 (micro): 0.6875
# Batch 131/805, Loss: 2.8131, F1 (weighted): 0.4375, F1 (macro): 0.2667, F1 (micro): 0.4375
# Batch 132/805, Loss: 2.9904, F1 (weighted): 0.2604, F1 (macro): 0.2193, F1 (micro): 0.3125
# Batch 133/805, Loss: 3.2013, F1 (weighted): 0.2083, F1 (macro): 0.1270, F1 (micro): 0.1875
# Batch 134/805, Loss: 2.7911, F1 (weighted): 0.2917, F1 (macro): 0.1667, F1 (micro): 0.3125
# Batch 135/805, Loss: 3.0236, F1 (weighted): 0.1875, F1 (macro): 0.1304, F1 (micro): 0.1875
# Batch 136/805, Loss: 2.8672, F1 (weighted): 0.2500, F1 (macro): 0.1667, F1 (micro): 0.2500
# Batch 137/805, Loss: 3.1266, F1 (weighted): 0.0833, F1 (macro): 0.0667, F1 (micro): 0.1250
# Batch 138/805, Loss: 2.8022, F1 (weighted): 0.1292, F1 (macro): 0.0939, F1 (micro): 0.1875
# Batch 139/805, Loss: 2.8363, F1 (weighted): 0.3021, F1 (macro): 0.1275, F1 (micro): 0.3125
# Batch 140/805, Loss: 3.0100, F1 (weighted): 0.1667, F1 (macro): 0.1333, F1 (micro): 0.1875
# Batch 141/805, Loss: 2.9071, F1 (weighted): 0.2604, F1 (macro): 0.1583, F1 (micro): 0.2500
# Batch 142/805, Loss: 2.6981, F1 (weighted): 0.3125, F1 (macro): 0.2778, F1 (micro): 0.3750
# Batch 143/805, Loss: 2.7978, F1 (weighted): 0.2292, F1 (macro): 0.1296, F1 (micro): 0.2500
# Batch 144/805, Loss: 2.8792, F1 (weighted): 0.1875, F1 (macro): 0.0952, F1 (micro): 0.1875
# Batch 145/805, Loss: 2.8584, F1 (weighted): 0.3917, F1 (macro): 0.2733, F1 (micro): 0.4375
# Batch 146/805, Loss: 2.7943, F1 (weighted): 0.2054, F1 (macro): 0.1729, F1 (micro): 0.2500
# Batch 147/805, Loss: 2.9101, F1 (weighted): 0.1042, F1 (macro): 0.0877, F1 (micro): 0.1250
# Batch 148/805, Loss: 2.8026, F1 (weighted): 0.3958, F1 (macro): 0.3235, F1 (micro): 0.4375
# Batch 149/805, Loss: 3.1177, F1 (weighted): 0.3333, F1 (macro): 0.2778, F1 (micro): 0.3750
# Batch 150/805, Loss: 2.5570, F1 (weighted): 0.6292, F1 (macro): 0.4980, F1 (micro): 0.6250
# Batch 151/805, Loss: 2.6960, F1 (weighted): 0.4583, F1 (macro): 0.2917, F1 (micro): 0.4375
# Batch 152/805, Loss: 2.9150, F1 (weighted): 0.1812, F1 (macro): 0.1450, F1 (micro): 0.2500
# Batch 153/805, Loss: 2.7152, F1 (weighted): 0.3292, F1 (macro): 0.2792, F1 (micro): 0.3750
# Batch 154/805, Loss: 2.8474, F1 (weighted): 0.2292, F1 (macro): 0.1228, F1 (micro): 0.1875
# Batch 155/805, Loss: 2.9553, F1 (weighted): 0.1875, F1 (macro): 0.1500, F1 (micro): 0.2500
# Batch 156/805, Loss: 2.4635, F1 (weighted): 0.5208, F1 (macro): 0.3725, F1 (micro): 0.5625
# Batch 157/805, Loss: 2.7771, F1 (weighted): 0.1667, F1 (macro): 0.1333, F1 (micro): 0.2500
# Batch 158/805, Loss: 2.7883, F1 (weighted): 0.3187, F1 (macro): 0.2263, F1 (micro): 0.3750
# Batch 159/805, Loss: 2.9249, F1 (weighted): 0.2750, F1 (macro): 0.1647, F1 (micro): 0.2500
# Batch 160/805, Loss: 2.6500, F1 (weighted): 0.3292, F1 (macro): 0.2481, F1 (micro): 0.3750
# Batch 161/805, Loss: 2.5301, F1 (weighted): 0.4167, F1 (macro): 0.2593, F1 (micro): 0.4375
# Batch 162/805, Loss: 2.7837, F1 (weighted): 0.1875, F1 (macro): 0.1228, F1 (micro): 0.2500
# Batch 163/805, Loss: 2.6033, F1 (weighted): 0.2771, F1 (macro): 0.1561, F1 (micro): 0.3125
# Batch 164/805, Loss: 2.9929, F1 (weighted): 0.2292, F1 (macro): 0.2037, F1 (micro): 0.2500
# Batch 165/805, Loss: 2.5138, F1 (weighted): 0.5000, F1 (macro): 0.3333, F1 (micro): 0.4375
# Batch 166/805, Loss: 2.8318, F1 (weighted): 0.3875, F1 (macro): 0.2421, F1 (micro): 0.4375
# Batch 167/805, Loss: 2.5397, F1 (weighted): 0.4646, F1 (macro): 0.4262, F1 (micro): 0.5000
# Batch 168/805, Loss: 2.6860, F1 (weighted): 0.1354, F1 (macro): 0.0758, F1 (micro): 0.1875
# Batch 169/805, Loss: 3.0106, F1 (weighted): 0.1607, F1 (macro): 0.0476, F1 (micro): 0.1875
# Batch 170/805, Loss: 2.5708, F1 (weighted): 0.1250, F1 (macro): 0.0417, F1 (micro): 0.1250
# Batch 171/805, Loss: 2.6674, F1 (weighted): 0.2083, F1 (macro): 0.1270, F1 (micro): 0.1875
# Batch 172/805, Loss: 2.9891, F1 (weighted): 0.1458, F1 (macro): 0.0972, F1 (micro): 0.1875
# Batch 173/805, Loss: 2.9596, F1 (weighted): 0.2083, F1 (macro): 0.1515, F1 (micro): 0.2500
# Batch 174/805, Loss: 2.9377, F1 (weighted): 0.2604, F1 (macro): 0.2193, F1 (micro): 0.3125
# Batch 175/805, Loss: 2.4521, F1 (weighted): 0.5208, F1 (macro): 0.3333, F1 (micro): 0.5625
# Batch 176/805, Loss: 2.4426, F1 (weighted): 0.4062, F1 (macro): 0.3056, F1 (micro): 0.4375
# Batch 177/805, Loss: 2.4457, F1 (weighted): 0.4062, F1 (macro): 0.2843, F1 (micro): 0.4375
# Batch 178/805, Loss: 2.5554, F1 (weighted): 0.3854, F1 (macro): 0.2333, F1 (micro): 0.3750
# Batch 179/805, Loss: 2.7462, F1 (weighted): 0.2917, F1 (macro): 0.2037, F1 (micro): 0.3125
# Batch 180/805, Loss: 2.6548, F1 (weighted): 0.3958, F1 (macro): 0.3016, F1 (micro): 0.4375
# Batch 181/805, Loss: 2.1116, F1 (weighted): 0.3958, F1 (macro): 0.3333, F1 (micro): 0.5000
# Batch 182/805, Loss: 2.7634, F1 (weighted): 0.2917, F1 (macro): 0.1746, F1 (micro): 0.3125
# Batch 183/805, Loss: 2.8621, F1 (weighted): 0.2188, F1 (macro): 0.1316, F1 (micro): 0.1875
# Batch 184/805, Loss: 2.5098, F1 (weighted): 0.4479, F1 (macro): 0.3611, F1 (micro): 0.5000
# Batch 185/805, Loss: 2.4011, F1 (weighted): 0.3229, F1 (macro): 0.2315, F1 (micro): 0.3750
# Batch 186/805, Loss: 2.8930, F1 (weighted): 0.5833, F1 (macro): 0.4178, F1 (micro): 0.5625
# Batch 187/805, Loss: 2.9156, F1 (weighted): 0.1250, F1 (macro): 0.0952, F1 (micro): 0.1250
# Batch 188/805, Loss: 2.7761, F1 (weighted): 0.2396, F1 (macro): 0.1863, F1 (micro): 0.2500
# Batch 189/805, Loss: 2.9433, F1 (weighted): 0.3375, F1 (macro): 0.2700, F1 (micro): 0.3750
# Batch 190/805, Loss: 2.8491, F1 (weighted): 0.1250, F1 (macro): 0.0909, F1 (micro): 0.1250
# Batch 191/805, Loss: 2.5173, F1 (weighted): 0.4851, F1 (macro): 0.2513, F1 (micro): 0.4375
# Batch 192/805, Loss: 2.4838, F1 (weighted): 0.3854, F1 (macro): 0.2895, F1 (micro): 0.4375
# Batch 193/805, Loss: 2.6339, F1 (weighted): 0.2604, F1 (macro): 0.1930, F1 (micro): 0.3125
# Batch 194/805, Loss: 2.3554, F1 (weighted): 0.3750, F1 (macro): 0.2549, F1 (micro): 0.3750
# Batch 195/805, Loss: 2.8132, F1 (weighted): 0.3125, F1 (macro): 0.1579, F1 (micro): 0.2500
# Batch 196/805, Loss: 2.4623, F1 (weighted): 0.3750, F1 (macro): 0.2708, F1 (micro): 0.3750
# Batch 197/805, Loss: 2.6138, F1 (weighted): 0.2875, F1 (macro): 0.1424, F1 (micro): 0.3125
# Batch 198/805, Loss: 2.4355, F1 (weighted): 0.2500, F1 (macro): 0.1364, F1 (micro): 0.3125
# Batch 199/805, Loss: 2.5279, F1 (weighted): 0.3750, F1 (macro): 0.2963, F1 (micro): 0.3750
# Batch 200/805, Loss: 2.8183, F1 (weighted): 0.2708, F1 (macro): 0.1508, F1 (micro): 0.2500
# Batch 201/805, Loss: 2.3319, F1 (weighted): 0.4062, F1 (macro): 0.3241, F1 (micro): 0.4375
# Batch 202/805, Loss: 3.0138, F1 (weighted): 0.2083, F1 (macro): 0.1349, F1 (micro): 0.2500
# Batch 203/805, Loss: 2.6647, F1 (weighted): 0.3542, F1 (macro): 0.2632, F1 (micro): 0.3750
# Batch 204/805, Loss: 2.6929, F1 (weighted): 0.2708, F1 (macro): 0.1970, F1 (micro): 0.3125
# Batch 205/805, Loss: 2.4678, F1 (weighted): 0.4062, F1 (macro): 0.2895, F1 (micro): 0.3750
# Batch 206/805, Loss: 2.8288, F1 (weighted): 0.2396, F1 (macro): 0.1863, F1 (micro): 0.2500
# Batch 207/805, Loss: 2.3269, F1 (weighted): 0.6607, F1 (macro): 0.5210, F1 (micro): 0.6875
# Batch 208/805, Loss: 2.5751, F1 (weighted): 0.3333, F1 (macro): 0.2000, F1 (micro): 0.3750
# Batch 209/805, Loss: 2.6619, F1 (weighted): 0.3854, F1 (macro): 0.2719, F1 (micro): 0.4375
# Batch 210/805, Loss: 2.6753, F1 (weighted): 0.1354, F1 (macro): 0.1083, F1 (micro): 0.1875
# Batch 211/805, Loss: 2.2999, F1 (weighted): 0.5521, F1 (macro): 0.3981, F1 (micro): 0.5625
# Batch 212/805, Loss: 2.6574, F1 (weighted): 0.2875, F1 (macro): 0.1900, F1 (micro): 0.3125
# Batch 213/805, Loss: 2.6426, F1 (weighted): 0.2917, F1 (macro): 0.1930, F1 (micro): 0.3125
# Batch 214/805, Loss: 2.4660, F1 (weighted): 0.4167, F1 (macro): 0.2632, F1 (micro): 0.4375
# Batch 215/805, Loss: 2.7402, F1 (weighted): 0.1688, F1 (macro): 0.0905, F1 (micro): 0.1875
# Batch 216/805, Loss: 2.5296, F1 (weighted): 0.5208, F1 (macro): 0.3833, F1 (micro): 0.5000
# Batch 217/805, Loss: 2.8797, F1 (weighted): 0.1042, F1 (macro): 0.0725, F1 (micro): 0.1250
# Batch 218/805, Loss: 2.3933, F1 (weighted): 0.4375, F1 (macro): 0.2667, F1 (micro): 0.4375
# Batch 219/805, Loss: 2.7192, F1 (weighted): 0.2125, F1 (macro): 0.1545, F1 (micro): 0.2500
# Batch 220/805, Loss: 2.4642, F1 (weighted): 0.3958, F1 (macro): 0.1930, F1 (micro): 0.3750
# Batch 221/805, Loss: 2.3933, F1 (weighted): 0.4167, F1 (macro): 0.2982, F1 (micro): 0.4375
# Batch 222/805, Loss: 2.4570, F1 (weighted): 0.3438, F1 (macro): 0.2685, F1 (micro): 0.3750
# Batch 223/805, Loss: 2.3905, F1 (weighted): 0.4792, F1 (macro): 0.4118, F1 (micro): 0.5000
# Batch 224/805, Loss: 2.5323, F1 (weighted): 0.3021, F1 (macro): 0.2451, F1 (micro): 0.3750
# Batch 225/805, Loss: 2.5252, F1 (weighted): 0.3333, F1 (macro): 0.2167, F1 (micro): 0.3750
# Batch 226/805, Loss: 2.7922, F1 (weighted): 0.1250, F1 (macro): 0.0580, F1 (micro): 0.1250
# Batch 227/805, Loss: 3.0027, F1 (weighted): 0.0729, F1 (macro): 0.0556, F1 (micro): 0.1250
# Batch 228/805, Loss: 2.3874, F1 (weighted): 0.3646, F1 (macro): 0.2870, F1 (micro): 0.4375
# Batch 229/805, Loss: 2.6146, F1 (weighted): 0.3958, F1 (macro): 0.2632, F1 (micro): 0.3750
# Batch 230/805, Loss: 2.6851, F1 (weighted): 0.2321, F1 (macro): 0.1353, F1 (micro): 0.2500
# Batch 231/805, Loss: 2.1817, F1 (weighted): 0.6250, F1 (macro): 0.4722, F1 (micro): 0.6250
# Batch 232/805, Loss: 2.5814, F1 (weighted): 0.1667, F1 (macro): 0.1333, F1 (micro): 0.1875
# Batch 233/805, Loss: 2.6726, F1 (weighted): 0.2500, F1 (macro): 0.1111, F1 (micro): 0.2500
# Batch 234/805, Loss: 2.5280, F1 (weighted): 0.2396, F1 (macro): 0.1250, F1 (micro): 0.2500
# Batch 235/805, Loss: 2.2458, F1 (weighted): 0.5583, F1 (macro): 0.3804, F1 (micro): 0.5625
# Batch 236/805, Loss: 2.4399, F1 (weighted): 0.5208, F1 (macro): 0.3509, F1 (micro): 0.5000
# Batch 237/805, Loss: 2.6430, F1 (weighted): 0.2292, F1 (macro): 0.1364, F1 (micro): 0.2500
# Batch 238/805, Loss: 2.3881, F1 (weighted): 0.5104, F1 (macro): 0.4479, F1 (micro): 0.5000
# Batch 239/805, Loss: 2.5461, F1 (weighted): 0.3125, F1 (macro): 0.2174, F1 (micro): 0.3125
# Batch 240/805, Loss: 2.4558, F1 (weighted): 0.3750, F1 (macro): 0.2500, F1 (micro): 0.3750
# Batch 241/805, Loss: 2.7179, F1 (weighted): 0.4062, F1 (macro): 0.2396, F1 (micro): 0.3750
# Batch 242/805, Loss: 2.5781, F1 (weighted): 0.3000, F1 (macro): 0.2196, F1 (micro): 0.3125
# Batch 243/805, Loss: 2.8108, F1 (weighted): 0.1458, F1 (macro): 0.1228, F1 (micro): 0.1875
# Batch 244/805, Loss: 2.4756, F1 (weighted): 0.3708, F1 (macro): 0.3310, F1 (micro): 0.4375
# Batch 245/805, Loss: 2.5120, F1 (weighted): 0.2708, F1 (macro): 0.2407, F1 (micro): 0.3125
# Batch 246/805, Loss: 2.4965, F1 (weighted): 0.3333, F1 (macro): 0.1833, F1 (micro): 0.3750
# Batch 247/805, Loss: 2.0097, F1 (weighted): 0.5167, F1 (macro): 0.3216, F1 (micro): 0.5625
# Batch 248/805, Loss: 2.3889, F1 (weighted): 0.4792, F1 (macro): 0.2963, F1 (micro): 0.4375
# Batch 249/805, Loss: 2.0455, F1 (weighted): 0.3958, F1 (macro): 0.2121, F1 (micro): 0.3750
# Batch 250/805, Loss: 2.5477, F1 (weighted): 0.2604, F1 (macro): 0.1842, F1 (micro): 0.3125
# Batch 251/805, Loss: 2.3930, F1 (weighted): 0.4375, F1 (macro): 0.3333, F1 (micro): 0.5000
# Batch 252/805, Loss: 2.2737, F1 (weighted): 0.3750, F1 (macro): 0.2105, F1 (micro): 0.3750
# Batch 253/805, Loss: 2.5876, F1 (weighted): 0.3083, F1 (macro): 0.1733, F1 (micro): 0.3125
# Batch 254/805, Loss: 2.5103, F1 (weighted): 0.4167, F1 (macro): 0.2857, F1 (micro): 0.4375
# Batch 255/805, Loss: 2.5799, F1 (weighted): 0.3500, F1 (macro): 0.2182, F1 (micro): 0.3750
# Batch 256/805, Loss: 1.9881, F1 (weighted): 0.7083, F1 (macro): 0.5333, F1 (micro): 0.6875
# Batch 257/805, Loss: 2.4231, F1 (weighted): 0.5208, F1 (macro): 0.4259, F1 (micro): 0.5625
# Batch 258/805, Loss: 2.2826, F1 (weighted): 0.5417, F1 (macro): 0.3750, F1 (micro): 0.5625
# Batch 259/805, Loss: 2.0550, F1 (weighted): 0.6833, F1 (macro): 0.4980, F1 (micro): 0.6875
# Batch 260/805, Loss: 2.3233, F1 (weighted): 0.5417, F1 (macro): 0.4298, F1 (micro): 0.5625
# Batch 261/805, Loss: 2.7917, F1 (weighted): 0.3417, F1 (macro): 0.1778, F1 (micro): 0.3125
# Batch 262/805, Loss: 2.1840, F1 (weighted): 0.6146, F1 (macro): 0.4537, F1 (micro): 0.5625
# Batch 263/805, Loss: 2.5295, F1 (weighted): 0.3021, F1 (macro): 0.2167, F1 (micro): 0.3750
# Batch 264/805, Loss: 2.2740, F1 (weighted): 0.4687, F1 (macro): 0.3796, F1 (micro): 0.5000
# Batch 265/805, Loss: 2.8734, F1 (weighted): 0.3958, F1 (macro): 0.2632, F1 (micro): 0.3750
# Batch 266/805, Loss: 2.5649, F1 (weighted): 0.2188, F1 (macro): 0.1190, F1 (micro): 0.1875
# Batch 267/805, Loss: 2.3114, F1 (weighted): 0.2500, F1 (macro): 0.2353, F1 (micro): 0.3125
# Batch 268/805, Loss: 2.3001, F1 (weighted): 0.4625, F1 (macro): 0.2762, F1 (micro): 0.4375
# Batch 269/805, Loss: 2.5350, F1 (weighted): 0.3333, F1 (macro): 0.2667, F1 (micro): 0.3750
# Batch 270/805, Loss: 2.2955, F1 (weighted): 0.4375, F1 (macro): 0.3500, F1 (micro): 0.5000
# Batch 271/805, Loss: 2.5869, F1 (weighted): 0.2188, F1 (macro): 0.1591, F1 (micro): 0.2500
# Batch 272/805, Loss: 2.5128, F1 (weighted): 0.2604, F1 (macro): 0.1842, F1 (micro): 0.3125
# Batch 273/805, Loss: 2.3480, F1 (weighted): 0.5417, F1 (macro): 0.3704, F1 (micro): 0.5625
# Batch 274/805, Loss: 2.6493, F1 (weighted): 0.4333, F1 (macro): 0.2627, F1 (micro): 0.4375
# Batch 275/805, Loss: 2.2247, F1 (weighted): 0.5625, F1 (macro): 0.4583, F1 (micro): 0.5625
# Batch 276/805, Loss: 2.4449, F1 (weighted): 0.3899, F1 (macro): 0.2381, F1 (micro): 0.4375
# Batch 277/805, Loss: 2.4329, F1 (weighted): 0.5149, F1 (macro): 0.3053, F1 (micro): 0.5000
# Batch 278/805, Loss: 2.3918, F1 (weighted): 0.4375, F1 (macro): 0.3431, F1 (micro): 0.4375
# Batch 279/805, Loss: 2.6199, F1 (weighted): 0.2500, F1 (macro): 0.1458, F1 (micro): 0.2500
# Batch 280/805, Loss: 2.4424, F1 (weighted): 0.3646, F1 (macro): 0.2917, F1 (micro): 0.4375
# Batch 281/805, Loss: 2.4353, F1 (weighted): 0.6667, F1 (macro): 0.5000, F1 (micro): 0.6875
# Batch 282/805, Loss: 2.0126, F1 (weighted): 0.6250, F1 (macro): 0.4902, F1 (micro): 0.6250
# Batch 283/805, Loss: 2.2990, F1 (weighted): 0.4583, F1 (macro): 0.3667, F1 (micro): 0.5000
# Batch 284/805, Loss: 2.3455, F1 (weighted): 0.4583, F1 (macro): 0.3667, F1 (micro): 0.5000
# Batch 285/805, Loss: 2.4174, F1 (weighted): 0.4062, F1 (macro): 0.2778, F1 (micro): 0.3750
# Batch 286/805, Loss: 2.1931, F1 (weighted): 0.4542, F1 (macro): 0.3593, F1 (micro): 0.5000
# Batch 287/805, Loss: 2.7746, F1 (weighted): 0.2875, F1 (macro): 0.1833, F1 (micro): 0.3125
# Batch 288/805, Loss: 2.3993, F1 (weighted): 0.3958, F1 (macro): 0.3241, F1 (micro): 0.4375
# Batch 289/805, Loss: 2.4719, F1 (weighted): 0.2812, F1 (macro): 0.2018, F1 (micro): 0.3125
# Batch 290/805, Loss: 2.1718, F1 (weighted): 0.3708, F1 (macro): 0.2702, F1 (micro): 0.4375
# Batch 291/805, Loss: 2.6982, F1 (weighted): 0.0833, F1 (macro): 0.0303, F1 (micro): 0.0625
# Batch 292/805, Loss: 2.4373, F1 (weighted): 0.5625, F1 (macro): 0.3889, F1 (micro): 0.5625
# Batch 293/805, Loss: 2.8615, F1 (weighted): 0.1562, F1 (macro): 0.0952, F1 (micro): 0.1875
# Batch 294/805, Loss: 2.4157, F1 (weighted): 0.4732, F1 (macro): 0.3083, F1 (micro): 0.5000
# Batch 295/805, Loss: 2.4997, F1 (weighted): 0.3125, F1 (macro): 0.1818, F1 (micro): 0.3125
# Batch 296/805, Loss: 2.1340, F1 (weighted): 0.5625, F1 (macro): 0.5333, F1 (micro): 0.6250
# Batch 297/805, Loss: 2.4360, F1 (weighted): 0.3125, F1 (macro): 0.1884, F1 (micro): 0.3125
# Batch 298/805, Loss: 2.3028, F1 (weighted): 0.5208, F1 (macro): 0.3684, F1 (micro): 0.5000
# Batch 299/805, Loss: 2.0855, F1 (weighted): 0.6250, F1 (macro): 0.4762, F1 (micro): 0.6875
# Batch 300/805, Loss: 2.5452, F1 (weighted): 0.4000, F1 (macro): 0.2175, F1 (micro): 0.3750
# Batch 301/805, Loss: 2.4633, F1 (weighted): 0.3021, F1 (macro): 0.1894, F1 (micro): 0.3125
# Batch 302/805, Loss: 2.5615, F1 (weighted): 0.2500, F1 (macro): 0.2000, F1 (micro): 0.2500
# Batch 303/805, Loss: 2.6994, F1 (weighted): 0.1875, F1 (macro): 0.1167, F1 (micro): 0.1875
# Batch 304/805, Loss: 2.5465, F1 (weighted): 0.3333, F1 (macro): 0.2963, F1 (micro): 0.3750
# Batch 305/805, Loss: 2.4366, F1 (weighted): 0.4062, F1 (macro): 0.2544, F1 (micro): 0.4375
# Batch 306/805, Loss: 2.1178, F1 (weighted): 0.5149, F1 (macro): 0.3838, F1 (micro): 0.5625
# Batch 307/805, Loss: 2.4142, F1 (weighted): 0.4583, F1 (macro): 0.3860, F1 (micro): 0.5000
# Batch 308/805, Loss: 2.5415, F1 (weighted): 0.2887, F1 (macro): 0.1399, F1 (micro): 0.2500
# Batch 309/805, Loss: 2.7478, F1 (weighted): 0.3958, F1 (macro): 0.2381, F1 (micro): 0.3750
# Batch 310/805, Loss: 2.5413, F1 (weighted): 0.2188, F1 (macro): 0.1842, F1 (micro): 0.2500
# Batch 311/805, Loss: 2.5600, F1 (weighted): 0.5208, F1 (macro): 0.3922, F1 (micro): 0.5625
# Batch 312/805, Loss: 2.5753, F1 (weighted): 0.4000, F1 (macro): 0.1727, F1 (micro): 0.3750
# Batch 313/805, Loss: 2.5630, F1 (weighted): 0.2708, F1 (macro): 0.1515, F1 (micro): 0.3125
# Batch 314/805, Loss: 2.3512, F1 (weighted): 0.2500, F1 (macro): 0.1739, F1 (micro): 0.2500
# Batch 315/805, Loss: 2.9689, F1 (weighted): 0.1979, F1 (macro): 0.1032, F1 (micro): 0.1875
# Batch 316/805, Loss: 2.3941, F1 (weighted): 0.3854, F1 (macro): 0.2833, F1 (micro): 0.4375
# Batch 317/805, Loss: 2.6238, F1 (weighted): 0.3542, F1 (macro): 0.2941, F1 (micro): 0.4375
# Batch 318/805, Loss: 2.3230, F1 (weighted): 0.5312, F1 (macro): 0.3421, F1 (micro): 0.5625
# Batch 319/805, Loss: 2.3557, F1 (weighted): 0.4062, F1 (macro): 0.2544, F1 (micro): 0.4375
# Batch 320/805, Loss: 2.3134, F1 (weighted): 0.5833, F1 (macro): 0.4259, F1 (micro): 0.5625
# Batch 321/805, Loss: 2.4726, F1 (weighted): 0.3750, F1 (macro): 0.2807, F1 (micro): 0.3750
# Batch 322/805, Loss: 2.3004, F1 (weighted): 0.3125, F1 (macro): 0.1667, F1 (micro): 0.3125
# Batch 323/805, Loss: 2.6326, F1 (weighted): 0.3542, F1 (macro): 0.2193, F1 (micro): 0.3750
# Batch 324/805, Loss: 2.3081, F1 (weighted): 0.4375, F1 (macro): 0.3333, F1 (micro): 0.4375
# Batch 325/805, Loss: 2.3029, F1 (weighted): 0.6083, F1 (macro): 0.3593, F1 (micro): 0.5625
# Batch 326/805, Loss: 2.4376, F1 (weighted): 0.5208, F1 (macro): 0.3509, F1 (micro): 0.5000
# Batch 327/805, Loss: 2.4846, F1 (weighted): 0.3125, F1 (macro): 0.2174, F1 (micro): 0.3125
# Batch 328/805, Loss: 2.0898, F1 (weighted): 0.4167, F1 (macro): 0.3148, F1 (micro): 0.4375
# Batch 329/805, Loss: 2.5572, F1 (weighted): 0.2083, F1 (macro): 0.1852, F1 (micro): 0.2500
# Batch 330/805, Loss: 2.6313, F1 (weighted): 0.3333, F1 (macro): 0.1833, F1 (micro): 0.3125
# Batch 331/805, Loss: 2.4170, F1 (weighted): 0.4375, F1 (macro): 0.2544, F1 (micro): 0.4375
# Batch 332/805, Loss: 2.1550, F1 (weighted): 0.4542, F1 (macro): 0.3037, F1 (micro): 0.5000
# Batch 333/805, Loss: 2.4348, F1 (weighted): 0.4062, F1 (macro): 0.2130, F1 (micro): 0.3750
# Batch 334/805, Loss: 2.5167, F1 (weighted): 0.3958, F1 (macro): 0.2879, F1 (micro): 0.4375
# Batch 335/805, Loss: 2.4209, F1 (weighted): 0.3083, F1 (macro): 0.2431, F1 (micro): 0.3750
# Batch 336/805, Loss: 2.2083, F1 (weighted): 0.6042, F1 (macro): 0.5333, F1 (micro): 0.6250
# Batch 337/805, Loss: 2.3480, F1 (weighted): 0.2771, F1 (macro): 0.1745, F1 (micro): 0.3125
# Batch 338/805, Loss: 2.5063, F1 (weighted): 0.2708, F1 (macro): 0.1583, F1 (micro): 0.2500
# Batch 339/805, Loss: 2.2333, F1 (weighted): 0.3125, F1 (macro): 0.1818, F1 (micro): 0.3750
# Batch 340/805, Loss: 2.0410, F1 (weighted): 0.7708, F1 (macro): 0.5926, F1 (micro): 0.7500
# Batch 341/805, Loss: 2.1211, F1 (weighted): 0.5208, F1 (macro): 0.3333, F1 (micro): 0.5000
# Batch 342/805, Loss: 2.4369, F1 (weighted): 0.4375, F1 (macro): 0.3039, F1 (micro): 0.4375
# Batch 343/805, Loss: 2.1503, F1 (weighted): 0.5000, F1 (macro): 0.4314, F1 (micro): 0.5625
# Batch 344/805, Loss: 2.3645, F1 (weighted): 0.4167, F1 (macro): 0.2833, F1 (micro): 0.4375
# Batch 345/805, Loss: 2.3169, F1 (weighted): 0.5625, F1 (macro): 0.3889, F1 (micro): 0.5625
# Batch 346/805, Loss: 2.6543, F1 (weighted): 0.2292, F1 (macro): 0.1833, F1 (micro): 0.2500
# Batch 347/805, Loss: 2.4479, F1 (weighted): 0.3750, F1 (macro): 0.2500, F1 (micro): 0.3750
# Batch 348/805, Loss: 2.3351, F1 (weighted): 0.5667, F1 (macro): 0.3228, F1 (micro): 0.5000
# Batch 349/805, Loss: 2.2796, F1 (weighted): 0.3958, F1 (macro): 0.2500, F1 (micro): 0.3750
# Batch 350/805, Loss: 2.6211, F1 (weighted): 0.3125, F1 (macro): 0.2000, F1 (micro): 0.3125
# Batch 351/805, Loss: 2.3671, F1 (weighted): 0.4792, F1 (macro): 0.4035, F1 (micro): 0.5000
# Batch 352/805, Loss: 2.0841, F1 (weighted): 0.5167, F1 (macro): 0.4042, F1 (micro): 0.5625
# Batch 353/805, Loss: 2.3527, F1 (weighted): 0.3292, F1 (macro): 0.2627, F1 (micro): 0.3750
# Batch 354/805, Loss: 2.4322, F1 (weighted): 0.5000, F1 (macro): 0.3860, F1 (micro): 0.5000
# Batch 355/805, Loss: 2.3988, F1 (weighted): 0.5250, F1 (macro): 0.2852, F1 (micro): 0.5000
# Batch 356/805, Loss: 2.4766, F1 (weighted): 0.2188, F1 (macro): 0.1842, F1 (micro): 0.2500
# Batch 357/805, Loss: 2.3563, F1 (weighted): 0.4375, F1 (macro): 0.3016, F1 (micro): 0.4375
# Batch 358/805, Loss: 2.3655, F1 (weighted): 0.3083, F1 (macro): 0.2175, F1 (micro): 0.3750
# Batch 359/805, Loss: 2.0912, F1 (weighted): 0.4688, F1 (macro): 0.2647, F1 (micro): 0.4375
# Batch 360/805, Loss: 2.2081, F1 (weighted): 0.3899, F1 (macro): 0.2513, F1 (micro): 0.4375
# Batch 361/805, Loss: 2.3609, F1 (weighted): 0.2812, F1 (macro): 0.1574, F1 (micro): 0.3125
# Batch 362/805, Loss: 2.4448, F1 (weighted): 0.4583, F1 (macro): 0.3030, F1 (micro): 0.4375
# Batch 363/805, Loss: 2.6547, F1 (weighted): 0.1667, F1 (macro): 0.0635, F1 (micro): 0.1250
# Batch 364/805, Loss: 2.3244, F1 (weighted): 0.5000, F1 (macro): 0.2807, F1 (micro): 0.5000
# Batch 365/805, Loss: 1.9522, F1 (weighted): 0.7750, F1 (macro): 0.6286, F1 (micro): 0.7500
# Batch 366/805, Loss: 2.4229, F1 (weighted): 0.3792, F1 (macro): 0.2204, F1 (micro): 0.3750
# Batch 367/805, Loss: 2.3456, F1 (weighted): 0.3604, F1 (macro): 0.2686, F1 (micro): 0.3750
# Batch 368/805, Loss: 2.2864, F1 (weighted): 0.3083, F1 (macro): 0.1733, F1 (micro): 0.3125
# Batch 369/805, Loss: 2.2485, F1 (weighted): 0.5208, F1 (macro): 0.4375, F1 (micro): 0.5625
# Batch 370/805, Loss: 2.3653, F1 (weighted): 0.1875, F1 (macro): 0.1316, F1 (micro): 0.1875
# Batch 371/805, Loss: 2.4462, F1 (weighted): 0.3542, F1 (macro): 0.2368, F1 (micro): 0.3750
# Batch 372/805, Loss: 2.3732, F1 (weighted): 0.3792, F1 (macro): 0.2627, F1 (micro): 0.3750
# Batch 373/805, Loss: 2.3314, F1 (weighted): 0.6000, F1 (macro): 0.4632, F1 (micro): 0.6250
# Batch 374/805, Loss: 2.6338, F1 (weighted): 0.2917, F1 (macro): 0.2333, F1 (micro): 0.3125
# Batch 375/805, Loss: 2.0405, F1 (weighted): 0.5938, F1 (macro): 0.3611, F1 (micro): 0.5625
# Batch 376/805, Loss: 2.3848, F1 (weighted): 0.4792, F1 (macro): 0.3833, F1 (micro): 0.5000
# Batch 377/805, Loss: 2.0346, F1 (weighted): 0.3363, F1 (macro): 0.2354, F1 (micro): 0.3750
# Batch 378/805, Loss: 2.5744, F1 (weighted): 0.2917, F1 (macro): 0.1894, F1 (micro): 0.3125
# Batch 379/805, Loss: 2.2426, F1 (weighted): 0.4375, F1 (macro): 0.2632, F1 (micro): 0.4375
# Batch 380/805, Loss: 2.3863, F1 (weighted): 0.4062, F1 (macro): 0.2396, F1 (micro): 0.3750
# Batch 381/805, Loss: 1.8112, F1 (weighted): 0.6458, F1 (macro): 0.5294, F1 (micro): 0.6250
# Batch 382/805, Loss: 2.3144, F1 (weighted): 0.4583, F1 (macro): 0.3667, F1 (micro): 0.5000
# Batch 383/805, Loss: 2.4858, F1 (weighted): 0.2292, F1 (macro): 0.1746, F1 (micro): 0.3125
# Batch 384/805, Loss: 2.2945, F1 (weighted): 0.4792, F1 (macro): 0.4167, F1 (micro): 0.5000
# Batch 385/805, Loss: 2.5096, F1 (weighted): 0.3500, F1 (macro): 0.2296, F1 (micro): 0.3750
# Batch 386/805, Loss: 2.1742, F1 (weighted): 0.5583, F1 (macro): 0.3778, F1 (micro): 0.5625
# Batch 387/805, Loss: 2.3740, F1 (weighted): 0.2917, F1 (macro): 0.2121, F1 (micro): 0.3125
# Batch 388/805, Loss: 2.4678, F1 (weighted): 0.4896, F1 (macro): 0.2870, F1 (micro): 0.4375
# Batch 389/805, Loss: 2.3917, F1 (weighted): 0.3958, F1 (macro): 0.2833, F1 (micro): 0.3750
# Batch 390/805, Loss: 2.3634, F1 (weighted): 0.3917, F1 (macro): 0.2296, F1 (micro): 0.3750
# Batch 391/805, Loss: 2.4808, F1 (weighted): 0.2604, F1 (macro): 0.1750, F1 (micro): 0.3125
# Batch 392/805, Loss: 2.0491, F1 (weighted): 0.6250, F1 (macro): 0.5000, F1 (micro): 0.6250
# Batch 393/805, Loss: 2.0760, F1 (weighted): 0.5000, F1 (macro): 0.3333, F1 (micro): 0.5000
# Batch 394/805, Loss: 2.1562, F1 (weighted): 0.7083, F1 (macro): 0.6000, F1 (micro): 0.6875
# Batch 395/805, Loss: 2.0689, F1 (weighted): 0.5729, F1 (macro): 0.5312, F1 (micro): 0.6250
# Batch 396/805, Loss: 2.2338, F1 (weighted): 0.4375, F1 (macro): 0.3750, F1 (micro): 0.4375
# Batch 397/805, Loss: 2.4475, F1 (weighted): 0.4375, F1 (macro): 0.2540, F1 (micro): 0.4375
# Batch 398/805, Loss: 2.5506, F1 (weighted): 0.3125, F1 (macro): 0.1667, F1 (micro): 0.3125
# Batch 399/805, Loss: 2.5615, F1 (weighted): 0.1875, F1 (macro): 0.1316, F1 (micro): 0.1875
# Batch 400/805, Loss: 2.3317, F1 (weighted): 0.4375, F1 (macro): 0.3000, F1 (micro): 0.4375
# Batch 401/805, Loss: 2.3270, F1 (weighted): 0.4375, F1 (macro): 0.3333, F1 (micro): 0.5000
# Batch 402/805, Loss: 2.3803, F1 (weighted): 0.2500, F1 (macro): 0.1364, F1 (micro): 0.2500
# Batch 403/805, Loss: 1.7818, F1 (weighted): 0.6417, F1 (macro): 0.5259, F1 (micro): 0.6875
# Batch 404/805, Loss: 2.1630, F1 (weighted): 0.4625, F1 (macro): 0.3053, F1 (micro): 0.5000
# Batch 405/805, Loss: 2.1726, F1 (weighted): 0.4667, F1 (macro): 0.3648, F1 (micro): 0.5000
# Batch 406/805, Loss: 2.2869, F1 (weighted): 0.5167, F1 (macro): 0.4311, F1 (micro): 0.5625
# Batch 407/805, Loss: 2.5752, F1 (weighted): 0.2812, F1 (macro): 0.2018, F1 (micro): 0.3125
# Batch 408/805, Loss: 2.5124, F1 (weighted): 0.3333, F1 (macro): 0.2667, F1 (micro): 0.3750
# Batch 409/805, Loss: 2.5114, F1 (weighted): 0.3542, F1 (macro): 0.2833, F1 (micro): 0.3750
# Batch 410/805, Loss: 2.2269, F1 (weighted): 0.6042, F1 (macro): 0.4118, F1 (micro): 0.6250
# Batch 411/805, Loss: 2.3288, F1 (weighted): 0.2708, F1 (macro): 0.1212, F1 (micro): 0.2500
# Batch 412/805, Loss: 1.7571, F1 (weighted): 0.7750, F1 (macro): 0.5500, F1 (micro): 0.7500
# Batch 413/805, Loss: 2.2919, F1 (weighted): 0.3167, F1 (macro): 0.2588, F1 (micro): 0.3750
# Batch 414/805, Loss: 2.3981, F1 (weighted): 0.2917, F1 (macro): 0.2222, F1 (micro): 0.3125
# Batch 415/805, Loss: 2.0279, F1 (weighted): 0.6458, F1 (macro): 0.4259, F1 (micro): 0.6250
# Batch 416/805, Loss: 2.7868, F1 (weighted): 0.2292, F1 (macro): 0.1429, F1 (micro): 0.2500
# Batch 417/805, Loss: 2.3932, F1 (weighted): 0.2396, F1 (macro): 0.2130, F1 (micro): 0.3125
# Batch 418/805, Loss: 2.3389, F1 (weighted): 0.5000, F1 (macro): 0.2593, F1 (micro): 0.5000
# Batch 419/805, Loss: 2.2198, F1 (weighted): 0.3958, F1 (macro): 0.2719, F1 (micro): 0.4375
# Batch 420/805, Loss: 2.3693, F1 (weighted): 0.2292, F1 (macro): 0.1746, F1 (micro): 0.2500
# Batch 421/805, Loss: 2.7993, F1 (weighted): 0.2708, F1 (macro): 0.1429, F1 (micro): 0.2500
# Batch 422/805, Loss: 2.5475, F1 (weighted): 0.4208, F1 (macro): 0.2127, F1 (micro): 0.3750
# Batch 423/805, Loss: 2.5696, F1 (weighted): 0.3125, F1 (macro): 0.2167, F1 (micro): 0.3125
# Batch 424/805, Loss: 2.2875, F1 (weighted): 0.2917, F1 (macro): 0.2121, F1 (micro): 0.3125
# Batch 425/805, Loss: 2.5871, F1 (weighted): 0.3333, F1 (macro): 0.2121, F1 (micro): 0.3125
# Batch 426/805, Loss: 2.5555, F1 (weighted): 0.2958, F1 (macro): 0.1649, F1 (micro): 0.3125
# Batch 427/805, Loss: 2.2767, F1 (weighted): 0.3646, F1 (macro): 0.3241, F1 (micro): 0.4375
# Batch 428/805, Loss: 2.0738, F1 (weighted): 0.5000, F1 (macro): 0.3684, F1 (micro): 0.5000
# Batch 429/805, Loss: 2.3633, F1 (weighted): 0.3229, F1 (macro): 0.3039, F1 (micro): 0.3750
# Batch 430/805, Loss: 2.3648, F1 (weighted): 0.4792, F1 (macro): 0.3833, F1 (micro): 0.5000
# Batch 431/805, Loss: 2.4106, F1 (weighted): 0.2917, F1 (macro): 0.1275, F1 (micro): 0.3125
# Batch 432/805, Loss: 2.4115, F1 (weighted): 0.3542, F1 (macro): 0.2315, F1 (micro): 0.3750
# Batch 433/805, Loss: 2.1255, F1 (weighted): 0.6250, F1 (macro): 0.4630, F1 (micro): 0.6250
# Batch 434/805, Loss: 2.3984, F1 (weighted): 0.4333, F1 (macro): 0.3037, F1 (micro): 0.4375
# Batch 435/805, Loss: 2.1167, F1 (weighted): 0.5417, F1 (macro): 0.4259, F1 (micro): 0.5625
# Batch 436/805, Loss: 2.2768, F1 (weighted): 0.3646, F1 (macro): 0.3241, F1 (micro): 0.4375
# Batch 437/805, Loss: 1.8942, F1 (weighted): 0.3839, F1 (macro): 0.3170, F1 (micro): 0.4375
# Batch 438/805, Loss: 1.9782, F1 (weighted): 0.4896, F1 (macro): 0.3889, F1 (micro): 0.5625
# Batch 439/805, Loss: 2.7083, F1 (weighted): 0.3021, F1 (macro): 0.1917, F1 (micro): 0.3125
# Batch 440/805, Loss: 2.2522, F1 (weighted): 0.4896, F1 (macro): 0.3438, F1 (micro): 0.5000
# Batch 441/805, Loss: 2.3719, F1 (weighted): 0.4792, F1 (macro): 0.3627, F1 (micro): 0.5000
# Batch 442/805, Loss: 2.6432, F1 (weighted): 0.3542, F1 (macro): 0.2368, F1 (micro): 0.3750
# Batch 443/805, Loss: 2.5646, F1 (weighted): 0.3125, F1 (macro): 0.2167, F1 (micro): 0.3125
# Batch 444/805, Loss: 2.5804, F1 (weighted): 0.2604, F1 (macro): 0.1894, F1 (micro): 0.3125
# Batch 445/805, Loss: 2.2145, F1 (weighted): 0.5375, F1 (macro): 0.3963, F1 (micro): 0.5625
# Batch 446/805, Loss: 2.3365, F1 (weighted): 0.4792, F1 (macro): 0.2982, F1 (micro): 0.4375
# Batch 447/805, Loss: 2.0958, F1 (weighted): 0.6042, F1 (macro): 0.5088, F1 (micro): 0.6250
# Batch 448/805, Loss: 2.4819, F1 (weighted): 0.3438, F1 (macro): 0.2250, F1 (micro): 0.3125
# Batch 449/805, Loss: 2.2102, F1 (weighted): 0.3542, F1 (macro): 0.2348, F1 (micro): 0.3750
# Batch 450/805, Loss: 2.0658, F1 (weighted): 0.4792, F1 (macro): 0.3651, F1 (micro): 0.5000
# Batch 451/805, Loss: 2.1926, F1 (weighted): 0.4583, F1 (macro): 0.2833, F1 (micro): 0.4375
# Batch 452/805, Loss: 2.4007, F1 (weighted): 0.5000, F1 (macro): 0.3529, F1 (micro): 0.5000
# Batch 453/805, Loss: 2.1823, F1 (weighted): 0.6250, F1 (macro): 0.4907, F1 (micro): 0.6250
# Batch 454/805, Loss: 2.3844, F1 (weighted): 0.5208, F1 (macro): 0.3333, F1 (micro): 0.5000
# Batch 455/805, Loss: 2.5544, F1 (weighted): 0.2917, F1 (macro): 0.2121, F1 (micro): 0.3125
# Batch 456/805, Loss: 2.1079, F1 (weighted): 0.4792, F1 (macro): 0.3333, F1 (micro): 0.5000
# Batch 457/805, Loss: 2.5934, F1 (weighted): 0.1875, F1 (macro): 0.1304, F1 (micro): 0.1875
# Batch 458/805, Loss: 2.2668, F1 (weighted): 0.5583, F1 (macro): 0.3417, F1 (micro): 0.5625
# Batch 459/805, Loss: 2.6699, F1 (weighted): 0.1875, F1 (macro): 0.1304, F1 (micro): 0.2500
# Batch 460/805, Loss: 2.0910, F1 (weighted): 0.6042, F1 (macro): 0.4825, F1 (micro): 0.6250
# Batch 461/805, Loss: 2.2163, F1 (weighted): 0.5167, F1 (macro): 0.4667, F1 (micro): 0.5625
# Batch 462/805, Loss: 2.5991, F1 (weighted): 0.3500, F1 (macro): 0.2526, F1 (micro): 0.3750
# Batch 463/805, Loss: 2.4313, F1 (weighted): 0.2083, F1 (macro): 0.0794, F1 (micro): 0.1875
# Batch 464/805, Loss: 2.4038, F1 (weighted): 0.5583, F1 (macro): 0.4148, F1 (micro): 0.5625
# Batch 465/805, Loss: 1.9534, F1 (weighted): 0.5625, F1 (macro): 0.5294, F1 (micro): 0.5625
# Batch 466/805, Loss: 2.0980, F1 (weighted): 0.6667, F1 (macro): 0.5556, F1 (micro): 0.6875
# Batch 467/805, Loss: 2.3661, F1 (weighted): 0.3125, F1 (macro): 0.2273, F1 (micro): 0.3125
# Batch 468/805, Loss: 2.3875, F1 (weighted): 0.2917, F1 (macro): 0.1746, F1 (micro): 0.3125
# Batch 469/805, Loss: 2.4818, F1 (weighted): 0.5417, F1 (macro): 0.3704, F1 (micro): 0.5000
# Batch 470/805, Loss: 2.2491, F1 (weighted): 0.5417, F1 (macro): 0.3137, F1 (micro): 0.5000
# Batch 471/805, Loss: 2.1799, F1 (weighted): 0.5417, F1 (macro): 0.2982, F1 (micro): 0.5625
# Batch 472/805, Loss: 2.3029, F1 (weighted): 0.4271, F1 (macro): 0.3137, F1 (micro): 0.5000
# Batch 473/805, Loss: 2.0421, F1 (weighted): 0.5521, F1 (macro): 0.4216, F1 (micro): 0.5625
# Batch 474/805, Loss: 2.4325, F1 (weighted): 0.5000, F1 (macro): 0.3860, F1 (micro): 0.5000
# Batch 475/805, Loss: 2.3209, F1 (weighted): 0.6250, F1 (macro): 0.4762, F1 (micro): 0.6250
# Batch 476/805, Loss: 2.4780, F1 (weighted): 0.3646, F1 (macro): 0.2193, F1 (micro): 0.3125
# Batch 477/805, Loss: 2.4202, F1 (weighted): 0.2396, F1 (macro): 0.1917, F1 (micro): 0.3125
# Batch 478/805, Loss: 2.3687, F1 (weighted): 0.3542, F1 (macro): 0.3125, F1 (micro): 0.3750
# Batch 479/805, Loss: 2.0587, F1 (weighted): 0.3646, F1 (macro): 0.3241, F1 (micro): 0.4375
# Batch 480/805, Loss: 2.3291, F1 (weighted): 0.4333, F1 (macro): 0.2877, F1 (micro): 0.4375
# Batch 481/805, Loss: 2.3994, F1 (weighted): 0.4375, F1 (macro): 0.3500, F1 (micro): 0.4375
# Batch 482/805, Loss: 2.3095, F1 (weighted): 0.2188, F1 (macro): 0.2059, F1 (micro): 0.3125
# Batch 483/805, Loss: 2.5587, F1 (weighted): 0.1458, F1 (macro): 0.1111, F1 (micro): 0.1875
# Batch 484/805, Loss: 1.9492, F1 (weighted): 0.6042, F1 (macro): 0.5476, F1 (micro): 0.6875
# Batch 485/805, Loss: 2.1812, F1 (weighted): 0.4167, F1 (macro): 0.2593, F1 (micro): 0.4375
# Batch 486/805, Loss: 2.1560, F1 (weighted): 0.5104, F1 (macro): 0.4510, F1 (micro): 0.5625
# Batch 487/805, Loss: 2.3239, F1 (weighted): 0.4167, F1 (macro): 0.3529, F1 (micro): 0.4375
# Batch 488/805, Loss: 2.7211, F1 (weighted): 0.2917, F1 (macro): 0.2456, F1 (micro): 0.3125
# Batch 489/805, Loss: 2.2929, F1 (weighted): 0.4167, F1 (macro): 0.2593, F1 (micro): 0.4375
# Batch 490/805, Loss: 2.2878, F1 (weighted): 0.3542, F1 (macro): 0.2833, F1 (micro): 0.3750
# Batch 491/805, Loss: 2.6208, F1 (weighted): 0.4375, F1 (macro): 0.3333, F1 (micro): 0.4375
# Batch 492/805, Loss: 2.5288, F1 (weighted): 0.1979, F1 (macro): 0.1032, F1 (micro): 0.1875
# Batch 493/805, Loss: 2.0732, F1 (weighted): 0.5583, F1 (macro): 0.4784, F1 (micro): 0.6250
# Batch 494/805, Loss: 2.4815, F1 (weighted): 0.2812, F1 (macro): 0.2647, F1 (micro): 0.3125
# Batch 495/805, Loss: 2.1731, F1 (weighted): 0.4104, F1 (macro): 0.2204, F1 (micro): 0.4375
# Batch 496/805, Loss: 2.1624, F1 (weighted): 0.4792, F1 (macro): 0.3083, F1 (micro): 0.5000
# Batch 497/805, Loss: 2.8153, F1 (weighted): 0.2708, F1 (macro): 0.1667, F1 (micro): 0.2500
# Batch 498/805, Loss: 2.7249, F1 (weighted): 0.3125, F1 (macro): 0.1587, F1 (micro): 0.3125
# Batch 499/805, Loss: 2.6708, F1 (weighted): 0.3125, F1 (macro): 0.1970, F1 (micro): 0.3125
# Batch 500/805, Loss: 2.3134, F1 (weighted): 0.3542, F1 (macro): 0.2456, F1 (micro): 0.3750
# Batch 501/805, Loss: 2.2281, F1 (weighted): 0.5208, F1 (macro): 0.3684, F1 (micro): 0.5000
# Batch 502/805, Loss: 2.0225, F1 (weighted): 0.5667, F1 (macro): 0.3216, F1 (micro): 0.5625
# Batch 503/805, Loss: 2.3878, F1 (weighted): 0.5000, F1 (macro): 0.3333, F1 (micro): 0.5000
# Batch 504/805, Loss: 2.2557, F1 (weighted): 0.4792, F1 (macro): 0.3438, F1 (micro): 0.5000
# Batch 505/805, Loss: 2.4019, F1 (weighted): 0.3438, F1 (macro): 0.2544, F1 (micro): 0.3750
# Batch 506/805, Loss: 2.3828, F1 (weighted): 0.2292, F1 (macro): 0.1667, F1 (micro): 0.2500
# Batch 507/805, Loss: 2.5056, F1 (weighted): 0.3125, F1 (macro): 0.2500, F1 (micro): 0.3125
# Batch 508/805, Loss: 2.3479, F1 (weighted): 0.3125, F1 (macro): 0.2632, F1 (micro): 0.3750
# Batch 509/805, Loss: 2.2564, F1 (weighted): 0.5417, F1 (macro): 0.4412, F1 (micro): 0.5625
# Batch 510/805, Loss: 2.0138, F1 (weighted): 0.6458, F1 (macro): 0.4706, F1 (micro): 0.6250
# Batch 511/805, Loss: 2.4285, F1 (weighted): 0.3125, F1 (macro): 0.1970, F1 (micro): 0.3125
# Batch 512/805, Loss: 1.9525, F1 (weighted): 0.5625, F1 (macro): 0.3333, F1 (micro): 0.5625
# Batch 513/805, Loss: 2.0541, F1 (weighted): 0.4583, F1 (macro): 0.3000, F1 (micro): 0.4375
# Batch 514/805, Loss: 2.0535, F1 (weighted): 0.3958, F1 (macro): 0.3333, F1 (micro): 0.4375
# Batch 515/805, Loss: 2.0730, F1 (weighted): 0.5625, F1 (macro): 0.4412, F1 (micro): 0.5625
# Batch 516/805, Loss: 2.3243, F1 (weighted): 0.5208, F1 (macro): 0.3860, F1 (micro): 0.5625
# Batch 517/805, Loss: 2.5113, F1 (weighted): 0.3333, F1 (macro): 0.2037, F1 (micro): 0.3125
# Batch 518/805, Loss: 2.2604, F1 (weighted): 0.4583, F1 (macro): 0.3725, F1 (micro): 0.5000
# Batch 519/805, Loss: 2.1770, F1 (weighted): 0.4708, F1 (macro): 0.3792, F1 (micro): 0.5000
# Batch 520/805, Loss: 2.3114, F1 (weighted): 0.3958, F1 (macro): 0.3070, F1 (micro): 0.4375
# Batch 521/805, Loss: 2.1771, F1 (weighted): 0.5458, F1 (macro): 0.3233, F1 (micro): 0.5000
# Batch 522/805, Loss: 2.2293, F1 (weighted): 0.4979, F1 (macro): 0.3648, F1 (micro): 0.5000
# Batch 523/805, Loss: 2.1433, F1 (weighted): 0.5625, F1 (macro): 0.3725, F1 (micro): 0.5625
# Batch 524/805, Loss: 2.5883, F1 (weighted): 0.2589, F1 (macro): 0.1880, F1 (micro): 0.3125
# Batch 525/805, Loss: 2.7287, F1 (weighted): 0.2292, F1 (macro): 0.1594, F1 (micro): 0.2500
# Batch 526/805, Loss: 1.9714, F1 (weighted): 0.5333, F1 (macro): 0.4333, F1 (micro): 0.6250
# Batch 527/805, Loss: 2.2057, F1 (weighted): 0.4583, F1 (macro): 0.3438, F1 (micro): 0.4375
# Batch 528/805, Loss: 2.1156, F1 (weighted): 0.5312, F1 (macro): 0.3796, F1 (micro): 0.5000
# Batch 529/805, Loss: 2.1339, F1 (weighted): 0.4479, F1 (macro): 0.2719, F1 (micro): 0.4375
# Batch 530/805, Loss: 2.2509, F1 (weighted): 0.3125, F1 (macro): 0.2273, F1 (micro): 0.3750
# Batch 531/805, Loss: 2.9902, F1 (weighted): 0.1458, F1 (macro): 0.1061, F1 (micro): 0.1875
# Batch 532/805, Loss: 2.2714, F1 (weighted): 0.5417, F1 (macro): 0.4074, F1 (micro): 0.5000
# Batch 533/805, Loss: 2.3140, F1 (weighted): 0.3958, F1 (macro): 0.3125, F1 (micro): 0.4375
# Batch 534/805, Loss: 1.9413, F1 (weighted): 0.7708, F1 (macro): 0.5417, F1 (micro): 0.7500
# Batch 535/805, Loss: 1.9861, F1 (weighted): 0.6399, F1 (macro): 0.4898, F1 (micro): 0.6875
# Batch 536/805, Loss: 2.0518, F1 (weighted): 0.5417, F1 (macro): 0.4259, F1 (micro): 0.5625
# Batch 537/805, Loss: 2.3507, F1 (weighted): 0.3854, F1 (macro): 0.2368, F1 (micro): 0.3750
# Batch 538/805, Loss: 2.4409, F1 (weighted): 0.5208, F1 (macro): 0.2982, F1 (micro): 0.5000
# Batch 539/805, Loss: 2.4329, F1 (weighted): 0.5000, F1 (macro): 0.3500, F1 (micro): 0.5000
# Batch 540/805, Loss: 2.4378, F1 (weighted): 0.4167, F1 (macro): 0.2982, F1 (micro): 0.4375
# Batch 541/805, Loss: 2.2733, F1 (weighted): 0.6083, F1 (macro): 0.5083, F1 (micro): 0.6250
# Batch 542/805, Loss: 2.2020, F1 (weighted): 0.5208, F1 (macro): 0.3889, F1 (micro): 0.5625
# Batch 543/805, Loss: 2.6265, F1 (weighted): 0.3542, F1 (macro): 0.2222, F1 (micro): 0.3750
# Batch 544/805, Loss: 2.4309, F1 (weighted): 0.3958, F1 (macro): 0.2870, F1 (micro): 0.3750
# Batch 545/805, Loss: 2.1174, F1 (weighted): 0.4333, F1 (macro): 0.3407, F1 (micro): 0.5000
# Batch 546/805, Loss: 2.2770, F1 (weighted): 0.6042, F1 (macro): 0.4510, F1 (micro): 0.6250
# Batch 547/805, Loss: 2.4578, F1 (weighted): 0.5625, F1 (macro): 0.3860, F1 (micro): 0.5625
# Batch 548/805, Loss: 2.1303, F1 (weighted): 0.3274, F1 (macro): 0.1855, F1 (micro): 0.3750
# Batch 549/805, Loss: 2.1790, F1 (weighted): 0.4792, F1 (macro): 0.3333, F1 (micro): 0.5000
# Batch 550/805, Loss: 2.1933, F1 (weighted): 0.4792, F1 (macro): 0.3158, F1 (micro): 0.5000
# Batch 551/805, Loss: 2.0424, F1 (weighted): 0.6292, F1 (macro): 0.4250, F1 (micro): 0.6250
# Batch 552/805, Loss: 2.3473, F1 (weighted): 0.3333, F1 (macro): 0.2544, F1 (micro): 0.3750
# Batch 553/805, Loss: 2.2805, F1 (weighted): 0.4583, F1 (macro): 0.2963, F1 (micro): 0.5000
# Batch 554/805, Loss: 2.2409, F1 (weighted): 0.5208, F1 (macro): 0.3529, F1 (micro): 0.5000
# Batch 555/805, Loss: 2.2844, F1 (weighted): 0.4375, F1 (macro): 0.3500, F1 (micro): 0.5000
# Batch 556/805, Loss: 2.4807, F1 (weighted): 0.3500, F1 (macro): 0.2526, F1 (micro): 0.3750
# Batch 557/805, Loss: 1.8955, F1 (weighted): 0.6250, F1 (macro): 0.4074, F1 (micro): 0.6250
# Batch 558/805, Loss: 2.2465, F1 (weighted): 0.4167, F1 (macro): 0.3426, F1 (micro): 0.4375
# Batch 559/805, Loss: 2.2379, F1 (weighted): 0.5000, F1 (macro): 0.3333, F1 (micro): 0.5000
# Batch 560/805, Loss: 2.4329, F1 (weighted): 0.3750, F1 (macro): 0.2083, F1 (micro): 0.3750
# Batch 561/805, Loss: 2.4289, F1 (weighted): 0.3125, F1 (macro): 0.2632, F1 (micro): 0.3125
# Batch 562/805, Loss: 2.2027, F1 (weighted): 0.5250, F1 (macro): 0.3407, F1 (micro): 0.5000
# Batch 563/805, Loss: 2.5352, F1 (weighted): 0.3125, F1 (macro): 0.2105, F1 (micro): 0.3125
# Batch 564/805, Loss: 2.3093, F1 (weighted): 0.5208, F1 (macro): 0.3981, F1 (micro): 0.5000
# Batch 565/805, Loss: 2.1805, F1 (weighted): 0.4750, F1 (macro): 0.3020, F1 (micro): 0.5000
# Batch 566/805, Loss: 2.3215, F1 (weighted): 0.4542, F1 (macro): 0.2877, F1 (micro): 0.5000
# Batch 567/805, Loss: 2.2480, F1 (weighted): 0.5417, F1 (macro): 0.3137, F1 (micro): 0.5000
# Batch 568/805, Loss: 1.9036, F1 (weighted): 0.6667, F1 (macro): 0.5444, F1 (micro): 0.6875
# Batch 569/805, Loss: 2.6060, F1 (weighted): 0.2250, F1 (macro): 0.1500, F1 (micro): 0.1875
# Batch 570/805, Loss: 2.4391, F1 (weighted): 0.2708, F1 (macro): 0.1667, F1 (micro): 0.2500
# Batch 571/805, Loss: 2.2744, F1 (weighted): 0.2812, F1 (macro): 0.1917, F1 (micro): 0.3125
# Batch 572/805, Loss: 2.1116, F1 (weighted): 0.6458, F1 (macro): 0.5185, F1 (micro): 0.6875
# Batch 573/805, Loss: 2.3865, F1 (weighted): 0.4583, F1 (macro): 0.2941, F1 (micro): 0.5000
# Batch 574/805, Loss: 2.3313, F1 (weighted): 0.2812, F1 (macro): 0.2045, F1 (micro): 0.3125
# Batch 575/805, Loss: 2.6163, F1 (weighted): 0.2917, F1 (macro): 0.2333, F1 (micro): 0.3750
# Batch 576/805, Loss: 2.3201, F1 (weighted): 0.3812, F1 (macro): 0.2529, F1 (micro): 0.4375
# Batch 577/805, Loss: 2.4802, F1 (weighted): 0.2917, F1 (macro): 0.2222, F1 (micro): 0.3125
# Batch 578/805, Loss: 2.5730, F1 (weighted): 0.2917, F1 (macro): 0.1905, F1 (micro): 0.3125
# Batch 579/805, Loss: 2.4009, F1 (weighted): 0.3750, F1 (macro): 0.2281, F1 (micro): 0.3750
# Batch 580/805, Loss: 2.5481, F1 (weighted): 0.3333, F1 (macro): 0.1825, F1 (micro): 0.3750
# Batch 581/805, Loss: 2.1940, F1 (weighted): 0.5417, F1 (macro): 0.4259, F1 (micro): 0.5625
# Batch 582/805, Loss: 2.3326, F1 (weighted): 0.4000, F1 (macro): 0.2784, F1 (micro): 0.4375
# Batch 583/805, Loss: 2.5630, F1 (weighted): 0.4542, F1 (macro): 0.2702, F1 (micro): 0.4375
# Batch 584/805, Loss: 2.3279, F1 (weighted): 0.3125, F1 (macro): 0.2778, F1 (micro): 0.3750
# Batch 585/805, Loss: 2.3307, F1 (weighted): 0.5208, F1 (macro): 0.2456, F1 (micro): 0.5000
# Batch 586/805, Loss: 2.3497, F1 (weighted): 0.4167, F1 (macro): 0.2963, F1 (micro): 0.4375
# Batch 587/805, Loss: 2.0562, F1 (weighted): 0.5000, F1 (macro): 0.3333, F1 (micro): 0.5000
# Batch 588/805, Loss: 2.7089, F1 (weighted): 0.2917, F1 (macro): 0.1746, F1 (micro): 0.3125
# Batch 589/805, Loss: 2.1727, F1 (weighted): 0.6667, F1 (macro): 0.5625, F1 (micro): 0.6875
# Batch 590/805, Loss: 2.4462, F1 (weighted): 0.3021, F1 (macro): 0.1944, F1 (micro): 0.3125
# Batch 591/805, Loss: 2.4708, F1 (weighted): 0.4375, F1 (macro): 0.2879, F1 (micro): 0.4375
# Batch 592/805, Loss: 2.2021, F1 (weighted): 0.4167, F1 (macro): 0.2833, F1 (micro): 0.4375
# Batch 593/805, Loss: 2.2635, F1 (weighted): 0.3958, F1 (macro): 0.3167, F1 (micro): 0.4375
# Batch 594/805, Loss: 2.3212, F1 (weighted): 0.5268, F1 (macro): 0.3112, F1 (micro): 0.5000
# Batch 595/805, Loss: 2.3113, F1 (weighted): 0.3708, F1 (macro): 0.2375, F1 (micro): 0.3750
# Batch 596/805, Loss: 2.1954, F1 (weighted): 0.4625, F1 (macro): 0.3895, F1 (micro): 0.5000
# Batch 597/805, Loss: 2.6939, F1 (weighted): 0.2083, F1 (macro): 0.1333, F1 (micro): 0.1875
# Batch 598/805, Loss: 2.6534, F1 (weighted): 0.4688, F1 (macro): 0.3070, F1 (micro): 0.4375
# Batch 599/805, Loss: 2.2657, F1 (weighted): 0.3958, F1 (macro): 0.2833, F1 (micro): 0.4375
# Batch 600/805, Loss: 2.4770, F1 (weighted): 0.3333, F1 (macro): 0.2593, F1 (micro): 0.3125
# Batch 601/805, Loss: 2.0825, F1 (weighted): 0.4958, F1 (macro): 0.2481, F1 (micro): 0.5000
# Batch 602/805, Loss: 2.1142, F1 (weighted): 0.5667, F1 (macro): 0.4311, F1 (micro): 0.5625
# Batch 603/805, Loss: 2.2474, F1 (weighted): 0.6042, F1 (macro): 0.4833, F1 (micro): 0.6250
# Batch 604/805, Loss: 2.2394, F1 (weighted): 0.5000, F1 (macro): 0.3667, F1 (micro): 0.5000
# Batch 605/805, Loss: 2.5330, F1 (weighted): 0.4583, F1 (macro): 0.3333, F1 (micro): 0.4375
# Batch 606/805, Loss: 2.2026, F1 (weighted): 0.5000, F1 (macro): 0.2941, F1 (micro): 0.4375
# Batch 607/805, Loss: 2.1157, F1 (weighted): 0.6875, F1 (macro): 0.5185, F1 (micro): 0.6875
# Batch 608/805, Loss: 2.2389, F1 (weighted): 0.5625, F1 (macro): 0.4444, F1 (micro): 0.6250
# Batch 609/805, Loss: 2.3973, F1 (weighted): 0.4417, F1 (macro): 0.2733, F1 (micro): 0.4375
# Batch 610/805, Loss: 2.4334, F1 (weighted): 0.3542, F1 (macro): 0.2281, F1 (micro): 0.3125
# Batch 611/805, Loss: 2.2775, F1 (weighted): 0.4688, F1 (macro): 0.3596, F1 (micro): 0.5000
# Batch 612/805, Loss: 2.5909, F1 (weighted): 0.2708, F1 (macro): 0.1250, F1 (micro): 0.2500
# Batch 613/805, Loss: 2.6560, F1 (weighted): 0.2708, F1 (macro): 0.1579, F1 (micro): 0.2500
# Batch 614/805, Loss: 2.2243, F1 (weighted): 0.3750, F1 (macro): 0.2807, F1 (micro): 0.3750
# Batch 615/805, Loss: 2.2218, F1 (weighted): 0.5000, F1 (macro): 0.4118, F1 (micro): 0.5625
# Batch 616/805, Loss: 2.4076, F1 (weighted): 0.5000, F1 (macro): 0.3333, F1 (micro): 0.5000
# Batch 617/805, Loss: 2.6724, F1 (weighted): 0.2604, F1 (macro): 0.2193, F1 (micro): 0.3125
# Batch 618/805, Loss: 2.1203, F1 (weighted): 0.5000, F1 (macro): 0.3667, F1 (micro): 0.5000
# Batch 619/805, Loss: 2.1708, F1 (weighted): 0.4375, F1 (macro): 0.2807, F1 (micro): 0.4375
# Batch 620/805, Loss: 2.2679, F1 (weighted): 0.5000, F1 (macro): 0.2778, F1 (micro): 0.5000
# Batch 621/805, Loss: 2.8153, F1 (weighted): 0.2667, F1 (macro): 0.1825, F1 (micro): 0.3125
# Batch 622/805, Loss: 2.2671, F1 (weighted): 0.5208, F1 (macro): 0.3235, F1 (micro): 0.5000
# Batch 623/805, Loss: 2.4500, F1 (weighted): 0.3500, F1 (macro): 0.2316, F1 (micro): 0.3125
# Batch 624/805, Loss: 2.2670, F1 (weighted): 0.3917, F1 (macro): 0.2733, F1 (micro): 0.4375
# Batch 625/805, Loss: 1.9900, F1 (weighted): 0.6354, F1 (macro): 0.5833, F1 (micro): 0.6250
# Batch 626/805, Loss: 2.1688, F1 (weighted): 0.6000, F1 (macro): 0.4867, F1 (micro): 0.6250
# Batch 627/805, Loss: 2.1908, F1 (weighted): 0.3542, F1 (macro): 0.2333, F1 (micro): 0.3750
# Batch 628/805, Loss: 2.2467, F1 (weighted): 0.4792, F1 (macro): 0.3651, F1 (micro): 0.5000
# Batch 629/805, Loss: 2.4983, F1 (weighted): 0.2667, F1 (macro): 0.1451, F1 (micro): 0.3125
# Batch 630/805, Loss: 2.3830, F1 (weighted): 0.2708, F1 (macro): 0.1930, F1 (micro): 0.2500
# Batch 631/805, Loss: 2.3179, F1 (weighted): 0.4062, F1 (macro): 0.3611, F1 (micro): 0.5000
# Batch 632/805, Loss: 2.4355, F1 (weighted): 0.4167, F1 (macro): 0.2833, F1 (micro): 0.4375
# Batch 633/805, Loss: 2.6241, F1 (weighted): 0.2500, F1 (macro): 0.0795, F1 (micro): 0.2500
# Batch 634/805, Loss: 2.7560, F1 (weighted): 0.2083, F1 (macro): 0.1667, F1 (micro): 0.2500
# Batch 635/805, Loss: 2.1181, F1 (weighted): 0.4792, F1 (macro): 0.3485, F1 (micro): 0.5000
# Batch 636/805, Loss: 2.2863, F1 (weighted): 0.4521, F1 (macro): 0.2614, F1 (micro): 0.4375
# Batch 637/805, Loss: 2.0935, F1 (weighted): 0.3750, F1 (macro): 0.2963, F1 (micro): 0.3750
# Batch 638/805, Loss: 2.0135, F1 (weighted): 0.5833, F1 (macro): 0.3958, F1 (micro): 0.5625
# Batch 639/805, Loss: 2.3481, F1 (weighted): 0.4167, F1 (macro): 0.2917, F1 (micro): 0.4375
# Batch 640/805, Loss: 2.0069, F1 (weighted): 0.4792, F1 (macro): 0.3500, F1 (micro): 0.5000
# Batch 641/805, Loss: 2.3637, F1 (weighted): 0.3917, F1 (macro): 0.2526, F1 (micro): 0.4375
# Batch 642/805, Loss: 2.2605, F1 (weighted): 0.5208, F1 (macro): 0.3039, F1 (micro): 0.5000
# Batch 643/805, Loss: 2.1278, F1 (weighted): 0.4688, F1 (macro): 0.3796, F1 (micro): 0.5000
# Batch 644/805, Loss: 2.5280, F1 (weighted): 0.5042, F1 (macro): 0.3037, F1 (micro): 0.5000
# Batch 645/805, Loss: 2.5336, F1 (weighted): 0.3750, F1 (macro): 0.2500, F1 (micro): 0.3750
# Batch 646/805, Loss: 2.0807, F1 (weighted): 0.5625, F1 (macro): 0.3889, F1 (micro): 0.5625
# Batch 647/805, Loss: 2.4486, F1 (weighted): 0.2292, F1 (macro): 0.1212, F1 (micro): 0.2500
# Batch 648/805, Loss: 2.0700, F1 (weighted): 0.6000, F1 (macro): 0.4458, F1 (micro): 0.6250
# Batch 649/805, Loss: 2.2778, F1 (weighted): 0.5000, F1 (macro): 0.3500, F1 (micro): 0.5000
# Batch 650/805, Loss: 2.2113, F1 (weighted): 0.3750, F1 (macro): 0.2167, F1 (micro): 0.3750
# Batch 651/805, Loss: 2.2735, F1 (weighted): 0.4167, F1 (macro): 0.2778, F1 (micro): 0.4375
# Batch 652/805, Loss: 2.0739, F1 (weighted): 0.5000, F1 (macro): 0.4118, F1 (micro): 0.5625
# Batch 653/805, Loss: 2.3765, F1 (weighted): 0.3500, F1 (macro): 0.2583, F1 (micro): 0.3750
# Batch 654/805, Loss: 2.8561, F1 (weighted): 0.0625, F1 (macro): 0.0476, F1 (micro): 0.0625
# Batch 655/805, Loss: 1.9329, F1 (weighted): 0.7292, F1 (macro): 0.5926, F1 (micro): 0.7500
# Batch 656/805, Loss: 2.1932, F1 (weighted): 0.4958, F1 (macro): 0.3315, F1 (micro): 0.5000
# Batch 657/805, Loss: 1.9665, F1 (weighted): 0.5208, F1 (macro): 0.3684, F1 (micro): 0.5000
# Batch 658/805, Loss: 2.1144, F1 (weighted): 0.5000, F1 (macro): 0.4444, F1 (micro): 0.5625
# Batch 659/805, Loss: 2.3269, F1 (weighted): 0.3917, F1 (macro): 0.2627, F1 (micro): 0.4375
# Batch 660/805, Loss: 2.2250, F1 (weighted): 0.5583, F1 (macro): 0.3870, F1 (micro): 0.5625
# Batch 661/805, Loss: 2.5450, F1 (weighted): 0.2917, F1 (macro): 0.1667, F1 (micro): 0.3125
# Batch 662/805, Loss: 2.1443, F1 (weighted): 0.4792, F1 (macro): 0.3725, F1 (micro): 0.5000
# Batch 663/805, Loss: 2.0876, F1 (weighted): 0.5062, F1 (macro): 0.3706, F1 (micro): 0.5625
# Batch 664/805, Loss: 2.3903, F1 (weighted): 0.4940, F1 (macro): 0.3069, F1 (micro): 0.5000
# Batch 665/805, Loss: 2.4310, F1 (weighted): 0.3021, F1 (macro): 0.1944, F1 (micro): 0.3125
# Batch 666/805, Loss: 2.3506, F1 (weighted): 0.6250, F1 (macro): 0.4737, F1 (micro): 0.6250
# Batch 667/805, Loss: 1.9815, F1 (weighted): 0.5667, F1 (macro): 0.3804, F1 (micro): 0.5625
# Batch 668/805, Loss: 2.4127, F1 (weighted): 0.2917, F1 (macro): 0.1833, F1 (micro): 0.3125
# Batch 669/805, Loss: 1.8731, F1 (weighted): 0.7917, F1 (macro): 0.7111, F1 (micro): 0.8125
# Batch 670/805, Loss: 2.4983, F1 (weighted): 0.2917, F1 (macro): 0.1833, F1 (micro): 0.3125
# Batch 671/805, Loss: 2.2057, F1 (weighted): 0.5792, F1 (macro): 0.4392, F1 (micro): 0.6250
# Batch 672/805, Loss: 2.3279, F1 (weighted): 0.5208, F1 (macro): 0.3704, F1 (micro): 0.5000
# Batch 673/805, Loss: 2.0724, F1 (weighted): 0.5750, F1 (macro): 0.4296, F1 (micro): 0.5625
# Batch 674/805, Loss: 2.2030, F1 (weighted): 0.5625, F1 (macro): 0.3968, F1 (micro): 0.5625
# Batch 675/805, Loss: 2.1728, F1 (weighted): 0.4583, F1 (macro): 0.4222, F1 (micro): 0.5000
# Batch 676/805, Loss: 2.1537, F1 (weighted): 0.3750, F1 (macro): 0.2083, F1 (micro): 0.3750
# Batch 677/805, Loss: 2.3153, F1 (weighted): 0.5208, F1 (macro): 0.3509, F1 (micro): 0.5000
# Batch 678/805, Loss: 1.9293, F1 (weighted): 0.5417, F1 (macro): 0.3651, F1 (micro): 0.5625
# Batch 679/805, Loss: 2.2607, F1 (weighted): 0.3750, F1 (macro): 0.2424, F1 (micro): 0.3750
# Batch 680/805, Loss: 2.7595, F1 (weighted): 0.2708, F1 (macro): 0.1159, F1 (micro): 0.2500
# Batch 681/805, Loss: 2.2518, F1 (weighted): 0.3542, F1 (macro): 0.2500, F1 (micro): 0.3750
# Batch 682/805, Loss: 2.4105, F1 (weighted): 0.4792, F1 (macro): 0.4062, F1 (micro): 0.5000
# Batch 683/805, Loss: 2.0860, F1 (weighted): 0.5625, F1 (macro): 0.4000, F1 (micro): 0.5625
# Batch 684/805, Loss: 2.1922, F1 (weighted): 0.4792, F1 (macro): 0.3529, F1 (micro): 0.5000
# Batch 685/805, Loss: 2.4055, F1 (weighted): 0.4688, F1 (macro): 0.3250, F1 (micro): 0.4375
# Batch 686/805, Loss: 2.2044, F1 (weighted): 0.4958, F1 (macro): 0.3593, F1 (micro): 0.5000
# Batch 687/805, Loss: 2.5866, F1 (weighted): 0.2604, F1 (macro): 0.2157, F1 (micro): 0.3125
# Batch 688/805, Loss: 2.5064, F1 (weighted): 0.1833, F1 (macro): 0.0815, F1 (micro): 0.1875
# Batch 689/805, Loss: 2.3767, F1 (weighted): 0.2604, F1 (macro): 0.2059, F1 (micro): 0.3125
# Batch 690/805, Loss: 2.6114, F1 (weighted): 0.3875, F1 (macro): 0.2706, F1 (micro): 0.4375
# Batch 691/805, Loss: 2.1337, F1 (weighted): 0.2917, F1 (macro): 0.1863, F1 (micro): 0.3125
# Batch 692/805, Loss: 2.5290, F1 (weighted): 0.4125, F1 (macro): 0.2762, F1 (micro): 0.4375
# Batch 693/805, Loss: 2.0225, F1 (weighted): 0.6726, F1 (macro): 0.5016, F1 (micro): 0.6875
# Batch 694/805, Loss: 2.2923, F1 (weighted): 0.3958, F1 (macro): 0.2010, F1 (micro): 0.3750
# Batch 695/805, Loss: 2.5386, F1 (weighted): 0.4062, F1 (macro): 0.3056, F1 (micro): 0.4375
# Batch 696/805, Loss: 2.4221, F1 (weighted): 0.2917, F1 (macro): 0.1818, F1 (micro): 0.3125
# Batch 697/805, Loss: 2.3966, F1 (weighted): 0.3958, F1 (macro): 0.2982, F1 (micro): 0.4375
# Batch 698/805, Loss: 2.1463, F1 (weighted): 0.3958, F1 (macro): 0.2667, F1 (micro): 0.4375
# Batch 699/805, Loss: 2.6681, F1 (weighted): 0.4211, F1 (macro): 0.2955, F1 (micro): 0.5000
# Batch 700/805, Loss: 2.1818, F1 (weighted): 0.3917, F1 (macro): 0.2733, F1 (micro): 0.4375
# Batch 701/805, Loss: 2.1460, F1 (weighted): 0.4792, F1 (macro): 0.4216, F1 (micro): 0.5000
# Batch 702/805, Loss: 2.5165, F1 (weighted): 0.3125, F1 (macro): 0.2167, F1 (micro): 0.3125
# Batch 703/805, Loss: 2.2418, F1 (weighted): 0.5625, F1 (macro): 0.4314, F1 (micro): 0.5625
# Batch 704/805, Loss: 2.5240, F1 (weighted): 0.2250, F1 (macro): 0.1556, F1 (micro): 0.3125
# Batch 705/805, Loss: 2.2439, F1 (weighted): 0.4167, F1 (macro): 0.3158, F1 (micro): 0.4375
# Batch 706/805, Loss: 2.8428, F1 (weighted): 0.2500, F1 (macro): 0.1364, F1 (micro): 0.2500
# Batch 707/805, Loss: 2.2211, F1 (weighted): 0.4167, F1 (macro): 0.2157, F1 (micro): 0.4375
# Batch 708/805, Loss: 2.4069, F1 (weighted): 0.3375, F1 (macro): 0.2000, F1 (micro): 0.3125
# Batch 709/805, Loss: 1.9733, F1 (weighted): 0.5000, F1 (macro): 0.3167, F1 (micro): 0.5000
# Batch 710/805, Loss: 2.3186, F1 (weighted): 0.3958, F1 (macro): 0.2037, F1 (micro): 0.3750
# Batch 711/805, Loss: 2.6250, F1 (weighted): 0.1696, F1 (macro): 0.0714, F1 (micro): 0.1875
# Batch 712/805, Loss: 2.2551, F1 (weighted): 0.3854, F1 (macro): 0.3426, F1 (micro): 0.4375
# Batch 713/805, Loss: 2.0582, F1 (weighted): 0.6042, F1 (macro): 0.4561, F1 (micro): 0.6250
# Batch 714/805, Loss: 2.6774, F1 (weighted): 0.1354, F1 (macro): 0.1083, F1 (micro): 0.1875
# Batch 715/805, Loss: 2.4957, F1 (weighted): 0.4375, F1 (macro): 0.2963, F1 (micro): 0.4375
# Batch 716/805, Loss: 1.9456, F1 (weighted): 0.5521, F1 (macro): 0.4222, F1 (micro): 0.5625
# Batch 717/805, Loss: 2.5127, F1 (weighted): 0.3438, F1 (macro): 0.1750, F1 (micro): 0.3125
# Batch 718/805, Loss: 2.2082, F1 (weighted): 0.6250, F1 (macro): 0.5000, F1 (micro): 0.6250
# Batch 719/805, Loss: 2.4829, F1 (weighted): 0.4375, F1 (macro): 0.2857, F1 (micro): 0.4375
# Batch 720/805, Loss: 2.2655, F1 (weighted): 0.5417, F1 (macro): 0.3684, F1 (micro): 0.5625
# Batch 721/805, Loss: 2.6272, F1 (weighted): 0.2812, F1 (macro): 0.2130, F1 (micro): 0.3125
# Batch 722/805, Loss: 2.2894, F1 (weighted): 0.2604, F1 (macro): 0.1894, F1 (micro): 0.3125
# Batch 723/805, Loss: 2.3663, F1 (weighted): 0.3125, F1 (macro): 0.1917, F1 (micro): 0.3125
# Batch 724/805, Loss: 2.4095, F1 (weighted): 0.3958, F1 (macro): 0.3148, F1 (micro): 0.4375
# Batch 725/805, Loss: 2.3931, F1 (weighted): 0.3125, F1 (macro): 0.2381, F1 (micro): 0.3125
# Batch 726/805, Loss: 2.5673, F1 (weighted): 0.3958, F1 (macro): 0.2381, F1 (micro): 0.3750
# Batch 727/805, Loss: 2.0737, F1 (weighted): 0.6042, F1 (macro): 0.5000, F1 (micro): 0.6250
# Batch 728/805, Loss: 2.2883, F1 (weighted): 0.3958, F1 (macro): 0.2333, F1 (micro): 0.3750
# Batch 729/805, Loss: 2.0285, F1 (weighted): 0.5833, F1 (macro): 0.3704, F1 (micro): 0.5625
# Batch 730/805, Loss: 2.3220, F1 (weighted): 0.4167, F1 (macro): 0.2456, F1 (micro): 0.3750
# Batch 731/805, Loss: 2.3750, F1 (weighted): 0.2083, F1 (macro): 0.1515, F1 (micro): 0.2500
# Batch 732/805, Loss: 2.4265, F1 (weighted): 0.4792, F1 (macro): 0.3333, F1 (micro): 0.4375
# Batch 733/805, Loss: 2.3253, F1 (weighted): 0.3542, F1 (macro): 0.2464, F1 (micro): 0.3750
# Batch 734/805, Loss: 2.0633, F1 (weighted): 0.4542, F1 (macro): 0.3593, F1 (micro): 0.5000
# Batch 735/805, Loss: 2.4506, F1 (weighted): 0.4750, F1 (macro): 0.3533, F1 (micro): 0.5000
# Batch 736/805, Loss: 2.4258, F1 (weighted): 0.2542, F1 (macro): 0.1298, F1 (micro): 0.2500
# Batch 737/805, Loss: 2.5792, F1 (weighted): 0.2083, F1 (macro): 0.1111, F1 (micro): 0.1875
# Batch 738/805, Loss: 2.5408, F1 (weighted): 0.3542, F1 (macro): 0.2464, F1 (micro): 0.3750
# Batch 739/805, Loss: 2.9036, F1 (weighted): 0.2708, F1 (macro): 0.1863, F1 (micro): 0.2500
# Batch 740/805, Loss: 2.3190, F1 (weighted): 0.2708, F1 (macro): 0.2063, F1 (micro): 0.3125
# Batch 741/805, Loss: 2.0477, F1 (weighted): 0.6417, F1 (macro): 0.4784, F1 (micro): 0.6250
# Batch 742/805, Loss: 2.3762, F1 (weighted): 0.3500, F1 (macro): 0.1912, F1 (micro): 0.3750
# Batch 743/805, Loss: 1.8525, F1 (weighted): 0.6458, F1 (macro): 0.3922, F1 (micro): 0.6250
# Batch 744/805, Loss: 2.2611, F1 (weighted): 0.5792, F1 (macro): 0.4392, F1 (micro): 0.6250
# Batch 745/805, Loss: 2.2068, F1 (weighted): 0.5000, F1 (macro): 0.3182, F1 (micro): 0.5000
# Batch 746/805, Loss: 2.6238, F1 (weighted): 0.2917, F1 (macro): 0.1833, F1 (micro): 0.3125
# Batch 747/805, Loss: 2.2132, F1 (weighted): 0.4958, F1 (macro): 0.3593, F1 (micro): 0.5000
# Batch 748/805, Loss: 2.2980, F1 (weighted): 0.3438, F1 (macro): 0.2549, F1 (micro): 0.3750
# Batch 749/805, Loss: 1.9984, F1 (weighted): 0.5625, F1 (macro): 0.4314, F1 (micro): 0.5625
# Batch 750/805, Loss: 2.4175, F1 (weighted): 0.4688, F1 (macro): 0.3824, F1 (micro): 0.5625
# Batch 751/805, Loss: 2.2251, F1 (weighted): 0.5000, F1 (macro): 0.3492, F1 (micro): 0.5000
# Batch 752/805, Loss: 2.1966, F1 (weighted): 0.4583, F1 (macro): 0.3492, F1 (micro): 0.5000
# Batch 753/805, Loss: 2.3127, F1 (weighted): 0.3438, F1 (macro): 0.2708, F1 (micro): 0.3750
# Batch 754/805, Loss: 2.3454, F1 (weighted): 0.3542, F1 (macro): 0.2500, F1 (micro): 0.3750
# Batch 755/805, Loss: 2.5837, F1 (weighted): 0.2917, F1 (macro): 0.1746, F1 (micro): 0.3125
# Batch 756/805, Loss: 2.1548, F1 (weighted): 0.4375, F1 (macro): 0.3333, F1 (micro): 0.4375
# Batch 757/805, Loss: 2.4382, F1 (weighted): 0.3958, F1 (macro): 0.2222, F1 (micro): 0.3750
# Batch 758/805, Loss: 2.4129, F1 (weighted): 0.1562, F1 (macro): 0.1250, F1 (micro): 0.1875
# Batch 759/805, Loss: 2.2081, F1 (weighted): 0.5417, F1 (macro): 0.4333, F1 (micro): 0.5625
# Batch 760/805, Loss: 2.6042, F1 (weighted): 0.2292, F1 (macro): 0.1159, F1 (micro): 0.2500
# Batch 761/805, Loss: 2.2500, F1 (weighted): 0.4375, F1 (macro): 0.2800, F1 (micro): 0.4375
# Batch 762/805, Loss: 2.3466, F1 (weighted): 0.4333, F1 (macro): 0.3000, F1 (micro): 0.4375
# Batch 763/805, Loss: 2.4397, F1 (weighted): 0.2708, F1 (macro): 0.2255, F1 (micro): 0.3125
# Batch 764/805, Loss: 2.2118, F1 (weighted): 0.6250, F1 (macro): 0.5098, F1 (micro): 0.6250
# Batch 765/805, Loss: 2.4228, F1 (weighted): 0.4792, F1 (macro): 0.3704, F1 (micro): 0.5000
# Batch 766/805, Loss: 2.1316, F1 (weighted): 0.6000, F1 (macro): 0.4000, F1 (micro): 0.5625
# Batch 767/805, Loss: 2.2787, F1 (weighted): 0.4542, F1 (macro): 0.3404, F1 (micro): 0.5000
# Batch 768/805, Loss: 2.5437, F1 (weighted): 0.2292, F1 (macro): 0.1746, F1 (micro): 0.2500
# Batch 769/805, Loss: 2.3984, F1 (weighted): 0.5250, F1 (macro): 0.3067, F1 (micro): 0.5000
# Batch 770/805, Loss: 2.3890, F1 (weighted): 0.4062, F1 (macro): 0.2685, F1 (micro): 0.3750
# Batch 771/805, Loss: 2.1371, F1 (weighted): 0.4375, F1 (macro): 0.3333, F1 (micro): 0.5000
# Batch 772/805, Loss: 2.0922, F1 (weighted): 0.4125, F1 (macro): 0.2762, F1 (micro): 0.4375
# Batch 773/805, Loss: 2.2640, F1 (weighted): 0.3292, F1 (macro): 0.1733, F1 (micro): 0.3750
# Batch 774/805, Loss: 2.7263, F1 (weighted): 0.1562, F1 (macro): 0.1190, F1 (micro): 0.1875
# Batch 775/805, Loss: 2.3800, F1 (weighted): 0.5417, F1 (macro): 0.4510, F1 (micro): 0.5625
# Batch 776/805, Loss: 2.6324, F1 (weighted): 0.3810, F1 (macro): 0.2073, F1 (micro): 0.3750
# Batch 777/805, Loss: 2.1377, F1 (weighted): 0.4375, F1 (macro): 0.3500, F1 (micro): 0.4375
# Batch 778/805, Loss: 2.2591, F1 (weighted): 0.5000, F1 (macro): 0.3860, F1 (micro): 0.5000
# Batch 779/805, Loss: 2.3931, F1 (weighted): 0.4375, F1 (macro): 0.3250, F1 (micro): 0.4375
# Batch 780/805, Loss: 2.1792, F1 (weighted): 0.5833, F1 (macro): 0.4278, F1 (micro): 0.5625
# Batch 781/805, Loss: 2.1516, F1 (weighted): 0.3854, F1 (macro): 0.3426, F1 (micro): 0.4375
# Batch 782/805, Loss: 2.1362, F1 (weighted): 0.4917, F1 (macro): 0.3917, F1 (micro): 0.5625
# Batch 783/805, Loss: 2.2097, F1 (weighted): 0.3333, F1 (macro): 0.2963, F1 (micro): 0.3750
# Batch 784/805, Loss: 2.4091, F1 (weighted): 0.2958, F1 (macro): 0.2140, F1 (micro): 0.3125
# Batch 785/805, Loss: 2.3019, F1 (weighted): 0.4375, F1 (macro): 0.2540, F1 (micro): 0.4375
# Batch 786/805, Loss: 2.3698, F1 (weighted): 0.5000, F1 (macro): 0.3182, F1 (micro): 0.5000
# Batch 787/805, Loss: 2.5388, F1 (weighted): 0.5417, F1 (macro): 0.3333, F1 (micro): 0.5000
# Batch 788/805, Loss: 2.5431, F1 (weighted): 0.3185, F1 (macro): 0.1262, F1 (micro): 0.3125
# Batch 789/805, Loss: 2.6201, F1 (weighted): 0.3125, F1 (macro): 0.2083, F1 (micro): 0.3125
# Batch 790/805, Loss: 2.1351, F1 (weighted): 0.5833, F1 (macro): 0.4035, F1 (micro): 0.5625
# Batch 791/805, Loss: 1.9353, F1 (weighted): 0.6667, F1 (macro): 0.4444, F1 (micro): 0.6875
# Batch 792/805, Loss: 2.2048, F1 (weighted): 0.4539, F1 (macro): 0.2515, F1 (micro): 0.4375
# Batch 793/805, Loss: 2.0936, F1 (weighted): 0.3750, F1 (macro): 0.2222, F1 (micro): 0.3750
# Batch 794/805, Loss: 2.3051, F1 (weighted): 0.5833, F1 (macro): 0.4333, F1 (micro): 0.5625
# Batch 795/805, Loss: 2.0969, F1 (weighted): 0.5625, F1 (macro): 0.3704, F1 (micro): 0.5625
# Batch 796/805, Loss: 2.3535, F1 (weighted): 0.3125, F1 (macro): 0.1368, F1 (micro): 0.3125
# Batch 797/805, Loss: 2.3910, F1 (weighted): 0.2979, F1 (macro): 0.2333, F1 (micro): 0.3750
# Batch 798/805, Loss: 2.3999, F1 (weighted): 0.5833, F1 (macro): 0.3704, F1 (micro): 0.5625
# Batch 799/805, Loss: 2.3675, F1 (weighted): 0.5417, F1 (macro): 0.4375, F1 (micro): 0.5625
# Batch 800/805, Loss: 2.5652, F1 (weighted): 0.4792, F1 (macro): 0.3529, F1 (micro): 0.5000
# Batch 801/805, Loss: 2.1712, F1 (weighted): 0.4542, F1 (macro): 0.4042, F1 (micro): 0.5000
# Batch 802/805, Loss: 2.0310, F1 (weighted): 0.5312, F1 (macro): 0.3796, F1 (micro): 0.5625
# Batch 803/805, Loss: 2.5606, F1 (weighted): 0.3125, F1 (macro): 0.1930, F1 (micro): 0.3125
# Batch 804/805, Loss: 2.1500, F1 (weighted): 0.4958, F1 (macro): 0.2481, F1 (micro): 0.5000
# Batch 805/805, Loss: 2.2144, F1 (weighted): 0.2500, F1 (macro): 0.1429, F1 (micro): 0.2500
# Validation F1 (weighted): 0.3993, Validation F1 (macro): 0.3897, Validation F1 (micro): 0.4524