import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from tqdm import tqdm
import numpy as np

# Настройки
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

# Загрузка данных
train_df = pd.read_csv('train.csv')  # Замените на путь к вашему тренировочному датасету
test_df = pd.read_csv('test.csv')    # Замените на путь к вашему тестовому датасету

# Токенизация данных
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')


class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = int(self.labels[idx])
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_LEN,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Подготовка данных для обучения и тестирования
train_texts, train_labels = train_df['text'], train_df['label']
test_texts, test_labels = test_df['text'], test_df['label']

# Создание модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=len(train_df['label'].unique()))
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)


# Функция обучения модели
def train_epoch(model, data_loader, optimizer, device):
    model.train()
    losses = []
    correct_predictions = 0

    for d in tqdm(data_loader):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return correct_predictions.double() / len(data_loader.dataset), sum(losses) / len(losses)


# Функция оценки модели
def eval_model(model, data_loader, device):
    model.eval()
    losses = []
    correct_predictions = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='weighted')
    roc_auc = roc_auc_score(all_labels, np.array(all_preds), multi_class='ovr')
    return correct_predictions.double() / len(data_loader.dataset), sum(losses) / len(losses), f1, roc_auc

# Кросс-валидация
kf = KFold(n_splits=5)

for fold, (train_index, val_index) in enumerate(kf.split(train_df)):
    print(f"Fold {fold + 1}")

    train_texts_fold, val_texts_fold = train_texts.iloc[train_index], train_texts.iloc[val_index]
    train_labels_fold, val_labels_fold = train_labels.iloc[train_index], train_labels.iloc[val_index]

    train_dataset_fold = TextDataset(train_texts_fold.values, train_labels_fold.values)
    val_dataset_fold = TextDataset(val_texts_fold.values, val_labels_fold.values)

    train_loader_fold = DataLoader(train_dataset_fold, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_fold = DataLoader(val_dataset_fold, batch_size=BATCH_SIZE)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        train_acc, train_loss = train_epoch(model, train_loader_fold, optimizer, device)
        print(f"Train loss {train_loss} accuracy {train_acc}")

        val_acc, val_loss, val_f1, val_roc_auc = eval_model(model, val_loader_fold, device)
        print(f"Validation loss {val_loss} accuracy {val_acc} F1 {val_f1} ROC AUC {val_roc_auc}")

# Оценка на тестовом наборе
test_dataset = TextDataset(test_texts.values, test_labels.values)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

test_acc, _, test_f1, test_roc_auc = eval_model(model, test_loader, device)
print(f'Test Accuracy: {test_acc} F1: {test_f1} ROC AUC: {test_roc_auc}')

# Вывод confusion matrix
all_preds_test = []
all_labels_test = []

model.eval()
with torch.no_grad():
    for d in test_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs.logits, dim=1)

        all_preds_test.extend(preds.cpu().numpy())
        all_labels_test.extend(labels.cpu().numpy())

conf_matrix = confusion_matrix(all_labels_test, all_preds_test)
print('Confusion Matrix:')
print(conf_matrix)
