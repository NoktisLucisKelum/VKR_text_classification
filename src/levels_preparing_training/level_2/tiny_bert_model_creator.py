import os
import string
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from transformers import AutoTokenizer, AutoModelForSequenceClassification

df = pd.read_csv(
    "/Users/denismazepa/Desktop/Py_projects/VKR/datasets/datasets_final/train_refactored_lematize_cut_final.csv",
    dtype={'RGNTI1': str, 'RGNTI2': str, 'RGNTI3': str})
old_unique_labels = df["RGNTI1"].unique().tolist()

columns = ["Index_name", "Name", "f1_weighted", "f1_micro", "f1_macro", "length", "unique_label"]
data = pd.DataFrame(columns=columns)

with open("/Users/denismazepa/Desktop/Py_projects/VKR/grnti/GRNTI_1_ru.json", 'r', encoding='utf-8') as f:
    json_data = json.load(f)

true_models_df = pd.read_excel("/Users/denismazepa/Desktop/Py_projects/VKR/src/levels_preparing_training/level_3/Results_level_3.xlsx",
                               dtype={'Index_name': str})

list_of_grnti_indexes = true_models_df["Index_name"].unique().tolist()


for i in old_unique_labels[2:7]:
    new_df = df[df["RGNTI1"] == i]

    # ----------------------------------------------------------------------------
    print("1. Разделение на train / val / test")
    # ----------------------------------------------------------------------------
    # train_df, test_df = train_test_split(
    #     df, test_size=0.15, random_state=42, stratify=df['label_id']
    # )
    new_df = new_df[new_df["RGNTI2"].isin(list_of_grnti_indexes)]
    print(list_of_grnti_indexes)
    unique_labels = sorted(new_df['RGNTI2'].unique().tolist())
    print(unique_labels)
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}

    new_df['label_id'] = new_df['RGNTI2'].apply(lambda x: label2id[x])
    train_df, val_df = train_test_split(
        new_df, test_size=0.1, random_state=42, stratify=new_df['label_id']
    )


    class TextDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len=64):
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
    print("3. Инициализация модели")
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

    print("4. Цикл обучения")

    epochs = 1  # Для примера

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
            print(
                f"Item: {i}, Batch {step + 1}/{len(train_loader)}, Loss: {loss.item():.4f}, F1 (weighted): {f1_weighted:.4f}, "
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
