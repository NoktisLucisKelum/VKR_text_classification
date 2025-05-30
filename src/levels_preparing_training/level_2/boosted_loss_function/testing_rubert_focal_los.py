import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import torch.nn.functional as F
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification

df = pd.read_csv(
    "/Users/denismazepa/Desktop/Py_projects/VKR/src/preprocessing/train_big_augmented_uncut_preprocessed_final_level_2_3.csv",
    dtype={'RGNTI1': str, 'RGNTI2': str, 'RGNTI3': str})
old_unique_labels = df["RGNTI1"].unique().tolist()
print(old_unique_labels)

columns = ["Index_name", "Name", "f1_weighted", "f1_micro", "f1_macro", "length", "unique_label"]
data = pd.DataFrame(columns=columns)

with open("/Users/denismazepa/Desktop/Py_projects/VKR/grnti/GRNTI_1_ru.json", 'r', encoding='utf-8') as f:
    json_data = json.load(f)

# true_models_df = pd.read_excel("/Users/denismazepa/Desktop/Py_projects/VKR/src/levels_preparing_training/level_3/Results_level_3.xlsx",
#                                dtype={'Index_name': str})
#
# list_of_grnti_indexes = true_models_df["Index_name"].unique().tolist()
#
#
# print(list_of_grnti_indexes)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        alpha (Tensor): Веса классов (размер [C]).
        gamma (float): Параметр фокусировки (чем больше, тем сильнее штрафуются простые примеры).
        reduction (str): 'mean', 'sum' или 'none'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # [B]
        pt = torch.exp(-ce_loss)  # p_t = вероятность правильного класса
        focal_loss = (1 - pt) ** self.gamma * ce_loss  # Focal Loss

        if self.alpha is not None:
            alpha = self.alpha[targets]  # Выбираем alpha для каждого элемента батча
            focal_loss = alpha * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


for i in old_unique_labels:
    new_df = df[df["RGNTI1"] == i]

    # ----------------------------------------------------------------------------
    print("1. Разделение на train / val / test")
    print(i, type(i))
    # new_df = new_df[new_df["RGNTI2"].isin(list_of_grnti_indexes)]
    unique_labels = sorted(new_df['RGNTI2'].unique().tolist())
    print(unique_labels)
    # if i == '75':
    #     print(f"{i}: точное попадание. Итоговый класс: 75.31.19")
    #     continue
    # ----------------------------------------------------------------------------
    # train_df, test_df = train_test_split(
    #     df, test_size=0.15, random_state=42, stratify=df['label_id']
    # )
    print(unique_labels)
    if len(unique_labels) == 1:
        data = data._append({
            "Index_name": i,
            "Name": json_data[i],
            "f1_weighted": 0,
            "f1_micro": 0,
            "f1_macro": 0,
            "length": 0,
            "unique_label": 1
        }, ignore_index=True)
        data.to_excel("Results_level_2.xlsx")
        continue

    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}

    new_df['label_id'] = new_df['RGNTI2'].apply(lambda x: label2id[x])
    train_df, val_df = train_test_split(
        new_df, test_size=0.1, random_state=42, stratify=new_df['label_id']
    )


    class TextDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len=100):
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
    print("2. Создаем датасеты и DataLoader-ы")
    # ----------------------------

    model_name = "sergeyzh/rubert-tiny-turbo"
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

    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ----------------------------
    print("3. Инициализация модели")
    # ----------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(unique_labels)  # важно указать, сколько у нас классов
    )
    class_counts = train_df['label_id'].value_counts().sort_index().values
    class_weights = 1.0 / np.sqrt(class_counts)  # Менее агрессивно, чем 1.0 / class_counts

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    weights_tensor = torch.FloatTensor(class_weights).to(device)

    # Перенесём на GPU при возможности

    # Определим оптимизатор и функцию потерь
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = FocalLoss(alpha=weights_tensor, gamma=2)

    print("4. Цикл обучения")

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
            logits = model(input_ids=input_ids,
                           attention_mask=attention_mask).logits

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * input_ids.size(0)

            # ---- Вычислим F1 (weighted) на текущем batch ----
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            gold = labels.cpu().numpy()
            f1_weighted = f1_score(gold, preds, average='weighted')
            f1_macro = f1_score(gold, preds, average='macro')
            f1_micro = f1_score(gold, preds, average='micro')

            # Выведем статистику
            if step % 70 == 2:
                print(
                    f"Item: {i}, Batch {step + 1}/{len(train_loader)}, Loss: {loss.item():.4f}, F1 (weighted): {f1_weighted:.4f}, "
                    f"F1 (macro): {f1_macro:.4f}, F1 (micro): {f1_micro:.4f}")

        # ---- Валидация на val_loader в конце эпохи (по желанию) ----
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                logits = model(input_ids=input_ids,
                               attention_mask=attention_mask).logits

                loss = criterion(logits, labels)
                val_loss += loss.item() * input_ids.size(0)

                preds = torch.argmax(logits, dim=1).cpu().numpy()
                gold = labels.cpu().numpy()

                val_preds.extend(preds)
                val_labels.extend(gold)

        val_f1_weighted = f1_score(val_labels, val_preds, average='weighted')
        val_f1_macro = f1_score(val_labels, val_preds, average='macro')
        val_f1_micro = f1_score(val_labels, val_preds, average='micro')
        if epoch + 1 == 1:
            data = data._append({
                "Index_name": i,
                "Name": json_data[i],
                "f1_weighted": val_f1_weighted,
                "f1_micro": val_f1_macro,
                "f1_macro": val_f1_micro,
                "length": len(train_df) + len(val_df),
                "unique_label": len(unique_labels)
            }, ignore_index=True)
            data.to_excel("Results_level_2.xlsx")

        print(f"Item: {i}, Validation F1 (weighted): {val_f1_weighted:.4f}, Validation F1 (macro): {val_f1_macro:.4f},"
              f" Validation F1 (micro): {val_f1_micro:.4f}")

    # ----------------------------
    print("5. Сохранение модели")
    # ----------------------------
    # Сохраним модель и токенайзер в папку rubert_tiny2_model
    save_path = f"{i}_model"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
