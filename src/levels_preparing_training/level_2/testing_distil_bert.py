import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, DistilBertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import numpy as np
from transformers import DistilBertTokenizer


def filter_common_values(df1, df2, column_name):
    # Получаем уникальные значения из указанного столбца для каждого DataFrame
    unique_values_df1 = set(df1[column_name].unique())
    unique_values_df2 = set(df2[column_name].unique())

    # Находим пересечение уникальных значений
    common_values = unique_values_df1.intersection(unique_values_df2)

    # Фильтруем каждый DataFrame, оставляя только те строки, где значения в столбце присутствуют в обоих DataFrame
    filtered_df1 = df1[df1[column_name].isin(common_values)].copy()
    filtered_df2 = df2[df2[column_name].isin(common_values)].copy()

    return filtered_df1, filtered_df2


train_df = pd.read_csv("/datasets/datasets_final/for_1_level/train_refactored_lematize_cut_final.csv",
                       dtype={'RGNTI1': str, 'RGNTI2': str, 'RGNTI3': str})
train_df = train_df[train_df['RGNTI1'] == "50"]
# print(train_df['RGNTI1'].value_counts())
valid_df = pd.read_csv("/datasets/datasets_final/for_1_level/test_refactored_lematize_cut_final.csv",
                       dtype={'RGNTI1': str, 'RGNTI2': str, 'RGNTI3': str})
valid_df = valid_df[valid_df['RGNTI1'] == "50"]

train_df, valid_df = filter_common_values(train_df, valid_df, 'RGNTI2')

print(len((train_df)), len(valid_df))
# print(valid_df['RGNTI1'].value_counts())
# test_df = pd.read_csv("/Users/denismazepa/Desktop/Py_projects/VKR/datasets/"
#                       "datasets_small/test_refactored_small.csv",
#                       dtype={'RGNTI1': str, 'RGNTI2': str, 'RGNTI3': str})
# print(test_df['RGNTI1'].value_counts())
label_encoder = LabelEncoder()
train_df['RGNTI2'] = label_encoder.fit_transform(train_df['RGNTI2'])
valid_df['RGNTI2'] = label_encoder.transform(valid_df['RGNTI2'])
print(1)
# Используем BertTokenizer для токенизации текстов
tokenizer = DistilBertTokenizer.from_pretrained('DmitryPogrebnoy/distilbert-base-russian-cased')

print(2)


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


print(3)


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = TextDataset(
        texts=df.body.to_numpy(),
        labels=df['RGNTI2'].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(ds, batch_size=batch_size)


BATCH_SIZE = 16
MAX_LEN = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 4
print(4)
train_data_loader = create_data_loader(train_df, tokenizer, MAX_LEN, BATCH_SIZE)
valid_data_loader = create_data_loader(valid_df, tokenizer, MAX_LEN, BATCH_SIZE)
# test_data_loader = create_data_loader(test_df, tokenizer, MAX_LEN, BATCH_SIZE, max_len=MAX_LEN)
print(5)
# Инициализируем модель BERT для классификации последовательностей
model = DistilBertForSequenceClassification.from_pretrained('DeepPavlov/rubert-base-cased',
                                                      num_labels=len(label_encoder.classes_))
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# Определяем функцию потерь
loss_fn = torch.nn.CrossEntropyLoss().to(device)
print(6)


# Функция для тренировки модели на одной эпохе


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler):
    model.train()
    losses = []
    correct_predictions = 0
    all_preds = []
    all_labels = []

    for d in tqdm(data_loader):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        _, preds = torch.max(outputs.logits, dim=1)
        loss = loss_fn(outputs.logits, labels)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        # Собираем все предсказания и метки
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # Вычисляем точность
    accuracy = correct_predictions.double() / len(data_loader.dataset)

    # Вычисляем F1-меру
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return accuracy, np.mean(losses), f1


print(7)


# Функция для оценки модели на валидационном наборе данных


def eval_model(model, data_loader, loss_fn, device):
    model.eval()
    losses = []
    correct_predictions = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, preds = torch.max(outputs.logits, dim=1)

            loss = loss_fn(outputs.logits, labels)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='weighted')
    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses), f1


print(8)

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_acc, train_loss, train_f1 = train_epoch(
        model,
        train_data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler
    )

    print(f'Train loss: {train_loss}, accuracy: {train_acc}, f1 {train_f1}')

    val_acc, val_loss, val_f1 = eval_model(
        model,
        valid_data_loader,
        loss_fn,
        device
    )

    print(f'Val loss: {val_loss}, accuracy: {val_acc}, f1: {val_f1}')

model.save_pretrained('bert_model')
tokenizer.save_pretrained('bert_model')

# Загрузка модели и токенизатора для предсказаний
model_loaded = DistilBertForSequenceClassification.from_pretrained('bert_model')
tokenizer_loaded = DistilBertTokenizer.from_pretrained('bert_model')

device = torch.device("cpu")


def predict_text(text):
    encoding = tokenizer_loaded.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    output = model_loaded(input_ids=input_ids, attention_mask=attention_mask)
    _, prediction = torch.max(output.logits, dim=1)

    return label_encoder.inverse_transform(prediction.cpu().numpy())


# Пример использования функции предсказания
sample_text = ("Алгоритм на базе пика плотности (DHeat) для кластеризации с эффективным радиусом	Кластеризация на базе "
               "плотности является одной из наиболее популярных парадигм среди существующих методов кластеризации, "
               "где большинство подходов такого типа, такие как DBSCAN, распознают кластеры данных, характеризуемые "
               "фиксированным радиусом сканирования. Однако существуют некоторые изъяны, вызванные фиксированным "
               "радиусом сканирования, напр., определение правильного радиуса сканирования является непростой задачей. ")
predicted_class = predict_text(sample_text)
print(f'Predicted class: {predicted_class}')
# Правильный ответ: 50.41