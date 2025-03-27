import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from equal_df import select_100_per_group
import pandas as pd
import json

# Допустим, что loaded_model, loaded_tokenizer, id2label и device уже инициализированы выше
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_path = "rubert_sentence_model"
loaded_tokenizer = AutoTokenizer.from_pretrained(save_path)
loaded_model = AutoModelForSequenceClassification.from_pretrained(save_path)
loaded_model.to(device)
loaded_model.eval()

with open('id2label.json', 'r', encoding='utf-8') as f:
    loaded_id2label = json.load(f)


def predict_class(text):
    """Функция предсказания классов на одном примере текста"""
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

        # Получаем индексы двух самых вероятных классов
        top_two_indices = np.argsort(probs)[-2:][::-1]

        # Получаем классы и их вероятности
        top_classes = [loaded_id2label[idx] for idx in top_two_indices.astype(str)]
        top_probabilities = probs[top_two_indices]
        # print(top_classes, top_probabilities)

    return top_classes, top_probabilities


# Тест на каком-то тексте
# top2_labels, top2_confidences = predict_class(test_text)
# print(f"Input_text: {test_text}")
test_text = input('Input text: here')
top2_labels, top2_confidences = predict_class(test_text)
print("Answer")
for i in range(len(top2_labels)):
    print(f"Class: {top2_labels[i]}")
    print(f"Probability: {round(float(top2_confidences[i]), 3)}")
# for label, conf in zip(top2_labels, top2_confidences):
#     print(f"Предсказанный класс: {label}, вероятность: {conf:.4f}")
