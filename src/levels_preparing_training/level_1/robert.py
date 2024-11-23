import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# Загрузка данных
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

# Преобразование текстовых меток в числовые
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_df['RGNTI1'])
val_labels = label_encoder.transform(valid_df['RGNTI1'])
# test_labels = label_encoder.transform(test_df['RGNTI1'])

# Подготовка токенизатора и модели
tokenizer = RobertaTokenizer.from_pretrained('blinoff/roberta-base-russian-v0')
model = RobertaForSequenceClassification.from_pretrained('blinoff/roberta-base-russian-v0', num_labels=len(label_encoder.classes_))


# Создание датасета
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
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


train_dataset = TextDataset(train_df['body'].tolist(), train_labels, tokenizer, 512)
val_dataset = TextDataset(valid_df['body'].tolist(), val_labels, tokenizer, 512)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Оптимизатор
optimizer = AdamW(model.parameters(), lr=2e-5)

# Обучение модели
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(3):  # Количество эпох можно изменить
    model.train()
    total_loss = 0
    total_accuracy = 0

    for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=-1)
        total_accuracy += accuracy_score(batch['labels'].cpu(), preds.cpu())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_loader)
    avg_train_accuracy = total_accuracy / len(train_loader)

    print(f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}')

    # Валидация модели
    model.eval()
    val_loss = 0
    val_accuracy = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            val_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            val_accuracy += accuracy_score(batch['labels'].cpu(), preds.cpu())

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    avg_val_accuracy = val_accuracy / len(val_loader)
    val_f1_score = f1_score(all_labels, all_preds, average='weighted')

    print(f'Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {avg_val_accuracy:.4f}, F1 Score: {val_f1_score:.4f}')

# Сохранение модели
model.save_pretrained('saved_model')

tokenizer.save_pretrained('saved_model')

# Загрузка модели и предсказание
loaded_model = RobertaForSequenceClassification.from_pretrained('saved_model')
loaded_tokenizer = RobertaTokenizer.from_pretrained('saved_model')
loaded_model.to(device)


def predict(text):
    inputs = loaded_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = loaded_model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=-1).item()
    return label_encoder.inverse_transform([predicted_class_id])[0]

# Пример использования функции предсказания
sample_text = "ваш текст здесь"
predicted_class = predict(sample_text)
print(f'Predicted class: {predicted_class}')