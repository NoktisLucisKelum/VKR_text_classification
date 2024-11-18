import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config, AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

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
test_labels = label_encoder.transform(test_df['RGNTI1'])

# Инициализация токенайзера
tokenizer = GPT2Tokenizer.from_pretrained('ai-forever/rugpt3small_based_on_gpt2')


# Создание датасета
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
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


# Создание DataLoader
def create_data_loader(df, tokenizer, max_len, batch_size):
    dataset = TextDataset(
        texts=df['body'].to_numpy(),
        labels=df['RGNTI1'].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(dataset, batch_size=batch_size)


BATCH_SIZE = 16

train_data_loader = create_data_loader(train_df, tokenizer, max_len=512, batch_size=BATCH_SIZE)
val_data_loader = create_data_loader(valid_df, tokenizer, max_len=512, batch_size=BATCH_SIZE)
test_data_loader = create_data_loader(test_df, tokenizer, max_len=512, batch_size=BATCH_SIZE)


# Модель
class TextClassificationModel(torch.nn.Module):
    def __init__(self, n_classes):
        super(TextClassificationModel, self).__init__()
        self.gpt2 = GPT2Model.from_pretrained('ai-forever/rugpt3small_based_on_gpt2')
        self.drop = torch.nn.Dropout(p=0.3)
        self.out = torch.nn.Linear(self.gpt2.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        last_hidden_state = self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state
        output = self.drop(last_hidden_state[:, -1, :])
        return self.out(output)


# Обучение модели
def train_epoch(
        model,
        data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        n_examples
):
    model = model.train()

    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask
                        )

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, labels)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)


# Основной цикл обучения
EPOCHS = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TextClassificationModel(len(label_encoder.classes_))
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=total_steps // 10, gamma=0.1)

loss_fn = torch.nn.CrossEntropyLoss().to(device)

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(
        model,
        train_data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        len(train_df)
    )

    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = eval_model(
        model,
        val_data_loader,
        loss_fn,
        device,
        len(valid_df)
    )

    print(f'Val   loss {val_loss} accuracy {val_acc}')

# Сохранение модели
torch.save(model.state_dict(), 'text_classification_model.bin')

# Загрузка модели
model.load_state_dict(torch.load('text_classification_model.bin'))


# Функция предсказания
def predict(text):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    output = model(input_ids, attention_mask)
    _, prediction = torch.max(output, dim=1)

    return label_encoder.inverse_transform(prediction.cpu().numpy())[0]


# Пример использования функции предсказания
sample_text = "ваш текст здесь"
predicted_class = predict(sample_text)
print(f'Predicted class: {predicted_class}')
