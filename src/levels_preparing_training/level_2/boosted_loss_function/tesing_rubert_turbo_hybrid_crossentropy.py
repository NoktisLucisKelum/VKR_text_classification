import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import torch.nn.functional as F
from sklearn.utils import compute_class_weight
from torch.optim.lr_scheduler import LinearLR
from transformers import AutoTokenizer, AutoModelForSequenceClassification

df = pd.read_csv(
    "/Users/denismazepa/Desktop/Py_projects/VKR/src/preprocessing/train_big_augmented_uncut_preprocessed_final_level_2_3.csv",
    dtype={'RGNTI1': str, 'RGNTI2': str, 'RGNTI3': str})
old_unique_labels = df["RGNTI1"].unique().tolist()
print(old_unique_labels)

columns = ["Index_name", "Name", "f1_weighted", "f1_micro", "f1_macro", "length", "unique_label"]
data = pd.DataFrame(columns=columns)

with open("/Users/denismazepa/Desktop/Py_projects/VKR/grnti/GRNTI_1_ru.json", 'r', encoding='utf-8') as f:
    data_json = json.load(f)

# print(list_of_grnti_indexes)
for top_idx in old_unique_labels:
    part_df = df[df["RGNTI1"] == top_idx]

    print('\n──────────', top_idx, '──────────')
    unique_labels = sorted(part_df["RGNTI2"].unique())
    if top_idx in ['34', '47', '06', '31']:
        print('75 → точное попадание, пропускаем')
        continue
    if len(unique_labels) == 1:                       # ничего обучать
        data = data._append({"Index_name": top_idx,
                             "Name": data_json[top_idx].replace(' ', '_'),
                             "f1_weighted": 0,
                             "f1_micro": 0,
                             "f1_macro": 0,
                             "length": 0,
                             "unique_label": 1},
                            ignore_index=True)
        data.to_excel("Results_level_2.xlsx")
        continue

    # -------------------------------------------------------
    # 1. маппинг меток и train/val split
    # -------------------------------------------------------
    label2id = {l: i for i, l in enumerate(unique_labels)}
    id2label = {i: l for l, i in label2id.items()}
    part_df["label_id"] = part_df["RGNTI2"].map(label2id)

    with open(f'{data_json[top_idx].replace(' ', '_')}.json', 'w', encoding='utf-8') as f:
        json.dump(id2label, f, ensure_ascii=False, indent=2)
    print(part_df.value_counts())

    train_df, val_df = train_test_split(
        part_df, test_size=0.1, random_state=42, stratify=part_df["label_id"]
    )

    # -------------------------------------------------------
    # 2. датасет
    # -------------------------------------------------------
    class TextDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len: int = 100):
            self.texts, self.labels = texts, labels
            self.tok, self.max_len = tokenizer, max_len

        def __len__(self):  return len(self.texts)

        def __getitem__(self, idx):
            enc = self.tok(self.texts[idx],
                           truncation=True,
                           padding='max_length',
                           max_length=self.max_len,
                           return_tensors='pt')
            item = {k: v.squeeze() for k, v in enc.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    model_name = "sergeyzh/rubert-tiny-turbo"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = TextDataset(train_df["body"].tolist(),
                           train_df["label_id"].tolist(),
                           tokenizer)
    val_ds = TextDataset(val_df["body"].tolist(),
                           val_df["label_id"].tolist(),
                           tokenizer)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds,   batch_size=8, shuffle=False)

    # -------------------------------------------------------
    # 3. модель
    # -------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForSequenceClassification.from_pretrained(
                 model_name, num_labels=len(unique_labels)).to(device)

    # -------------------------------------------------------
    # 4. ВЕСА ДЛЯ КРОСС‑ЭНТРОПИИ
    # -------------------------------------------------------
    #   weight_c[i] = 1 / freq(i)
    value_counts = part_df["RGNTI2"].value_counts()
    most_common_val = value_counts.idxmax()
    most_common_count = value_counts.max()
    ratio = most_common_count / len(part_df)

    freq = torch.tensor(
        train_df['label_id'].value_counts().sort_index().values,
        dtype=torch.float)

    # 2. Мягкие веса ---------------------------------------------------------
    # base = mean_freq / freq  ≈ 1/freq, но приведённые к среднему значению 1
    base_w = freq.mean() / freq

    if ratio >= 0.5:
       alpha = 0.45 # 0.25–0.75 → чем меньше, тем мягче
    elif 0.45 <= ratio < 0.5:
        alpha = 0.35
    elif 0.4 <= ratio < 0.45:
        alpha = 0.3
    elif 0.1 <= ratio < 0.3:
        alpha = 0.2
    else:
        alpha = 0.15
    print(ratio, alpha)
    weights = base_w.pow(alpha)  # (mean/freq)^α

    # 3. Ограничим разлёт (опционально)
    weights = torch.clamp(weights, min=0.2, max=5.0)

    # 4. Нормируем, чтобы среднее ≈ 1  (чисто косметика: learning‑rate не «прыгает»)
    weights = weights / weights.mean()
    initial_lr = 2e-4  # Стартовый LR (например, как у BERT)

    # 5. Функция потерь
    loss_fn = nn.CrossEntropyLoss(
        weight=weights.to(device),
        label_smoothing=0.05)  # слегка разглаживает целевые one‑hot
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr)

    epochs = 2
    total_steps = len(train_loader) * epochs
    final_lr = 7e-5  # Минимальный LR
    if 6000 < len(part_df) < 10000:
        final_div_factor = 18
        pct_start=0.4
    else:
        final_div_factor = 10
        pct_start = 0.3
    scheduler = OneCycleLR(
        optimizer,
        max_lr=initial_lr,  # Пиковый LR
        total_steps=total_steps,  # Общее число шагов
        pct_start=0.4,  # 30% шагов — рост LR
        anneal_strategy="linear",  # Линейное снижение
        final_div_factor=18,  # final_lr = max_lr / 10
    )

    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch + 1}/{epochs} ===")

        # ---------- TRAIN ----------
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)

            logits = outputs.logits
            loss = loss_fn(logits, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item()

            # ----- F1 по текущему batch'у -----
            preds = torch.argmax(logits, dim=1).cpu().tolist()  # <- tolist()
            gold = labels.cpu().tolist()

            f1_weighted = f1_score(gold, preds, average='weighted')
            f1_macro = f1_score(gold, preds, average='macro')
            f1_micro = f1_score(gold, preds, average='micro')
            if step % 60 == 1:
                print(f"Item: {top_idx}, Batch {step + 1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, "
                      f"F1 (weighted/macro/micro): "
                      f"{f1_weighted:.4f}/{f1_macro:.4f}/{f1_micro:.4f}")

        # ---------- VALIDATION ----------
        model.eval()
        val_preds, val_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                logits = model(input_ids=input_ids,
                               attention_mask=attention_mask).logits

                preds = torch.argmax(logits, dim=1).cpu().tolist()  # <- tolist()
                gold = labels.cpu().tolist()

                # extend плоскими списками int‑ов
                val_preds.extend(preds)
                val_labels.extend(gold)

        # теперь оба списка точно одномерные и одинаковой длины
        val_f1_weighted = f1_score(val_labels, val_preds, average='weighted')
        val_f1_macro = f1_score(val_labels, val_preds, average='macro')
        val_f1_micro = f1_score(val_labels, val_preds, average='micro')

        if epoch == 2:
            data = data._append(
                {"Index_name": top_idx,
                 "Name": data_json[top_idx].replace(' ', '_'),
                 "f1_weighted": val_f1_weighted,
                 "f1_macro": val_f1_macro,
                 "f1_micro": val_f1_micro,  # исправил порядок
                 "length": len(train_df) + len(val_df),
                 "unique_label": len(unique_labels)},
                ignore_index=True)
            data.to_excel("Results_level_2.xlsx", index=False)

        print(f"Item: {top_idx}, Validation F1 (weighted/macro/micro): "
              f"{val_f1_weighted:.4f}/{val_f1_macro:.4f}/{val_f1_micro:.4f}")

    # ---------- SAVE ----------
    save_path = f"./{top_idx}_model"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)