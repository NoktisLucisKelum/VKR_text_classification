import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 5
# Загружаем данные
train_df = pd.read_csv("/Users/denismazepa/Desktop/Py_projects/VKR/"
                       "datasets/datasets_small/train_refactored_small.csv",
                       dtype={'RGNTI1': str, 'RGNTI2': str, 'RGNTI3': str})
print(train_df['RGNTI1'].value_counts())
valid_df = pd.read_csv("/Users/denismazepa/Desktop/Py_projects/VKR/datasets/"
                       "datasets_small/train_refactored_small_validation.csv",
                        dtype={'RGNTI1': str, 'RGNTI2': str, 'RGNTI3': str})
print(valid_df['RGNTI1'].value_counts())
test_df = pd.read_csv("/Users/denismazepa/Desktop/Py_projects/VKR/datasets/"
                      "datasets_small/test_refactored_small.csv",
                      dtype={'RGNTI1': str, 'RGNTI2': str, 'RGNTI3': str})
print(test_df['RGNTI1'].value_counts())

# Преобразуем текстовые классы в числовые
label_encoder = LabelEncoder()
train_df['RGNTI1'] = label_encoder.fit_transform(train_df['RGNTI1'])
valid_df['RGNTI1'] = label_encoder.transform(valid_df['RGNTI1'])
test_df['RGNTI1'] = label_encoder.transform(test_df['RGNTI1'])

# Используем BertTokenizer для токенизации текстов
tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')


# Создаем кастомный датасет

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
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


# Создаем DataLoader для каждого набора данных
def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = TextDataset(
        texts=df.body.to_numpy(),
        labels=df['RGNTI1'].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(ds, batch_size=batch_size)


BATCH_SIZE = 16
MAX_LEN = 512

train_data_loader = create_data_loader(train_df, tokenizer, MAX_LEN, BATCH_SIZE)
valid_data_loader = create_data_loader(valid_df, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(test_df, tokenizer, MAX_LEN, BATCH_SIZE)

# Инициализируем модель BERT для классификации последовательностей
model = BertForSequenceClassification.from_pretrained('DeepPavlov/rubert-base-cased',
                                                      num_labels=len(label_encoder.classes_))
model = model.to(device)

# Определяем оптимизатор и планировщик обучения
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# Определяем функцию потерь
loss_fn = torch.nn.CrossEntropyLoss().to(device)


# Функция для тренировки модели на одной эпохе
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler):
    model.train()
    losses = []
    correct_predictions = 0

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

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)


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


for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(
        model,
        train_data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler
    )

    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss, val_f1 = eval_model(
        model,
        valid_data_loader,
        loss_fn,
        device
    )

    print(f'Val   loss {val_loss} accuracy {val_acc} f1 {val_f1}')

# Сохраняем модель
model.save_pretrained('/Users/denismazepa/Desktop/Py_projects/VKR/models/bert_model')
tokenizer.save_pretrained('/Users/denismazepa/Desktop/Py_projects/VKR/models/bert_model')

# Загрузка модели и токенизатора для предсказаний
model_loaded = BertForSequenceClassification.from_pretrained('../../../models/bert_model')
tokenizer_loaded = BertTokenizer.from_pretrained('../../../models/bert_model')


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


sample_text = "ваш текст здесь"
predicted_class = predict_text(sample_text)
print(f'Predicted class: {predicted_class}')
