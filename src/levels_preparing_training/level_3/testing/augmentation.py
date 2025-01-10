import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from time import time

model_name = "ai-forever/rugpt3medium_based_on_gpt2"
print(0)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
print(2)
model = GPT2LMHeadModel.from_pretrained(model_name)
print(3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def augment_dataset_with_rugpt3(dataset, text_column, class_column):
    """
    Увеличивает размер датасета, если его длина меньше 1000, добавляя новые строки с перефразированными текстами.

    :param dataset: pandas DataFrame с текстами и классами
    :param text_column: Название столбца с текстами
    :param class_column: Название столбца с классами
    :return: Обновленный DataFrame
    """

    dataset_length = len(dataset)
    unique_classes = dataset[class_column].nunique()
    class_counts = dataset[class_column].value_counts()

    print(f"Длина датасета: {dataset_length}")
    print(f"Количество уникальных классов: {unique_classes}")
    print(f"Количество значений для каждого класса:\n{class_counts}")
    if dataset_length >= 1000:
        print("Датасет содержит 1000 или более строк. Увеличение не требуется.")
        return dataset
    print("Увеличиваем размер датасета...")
    augmented_data = []

    for cls in class_counts.index:
        class_texts = dataset[dataset[class_column] == cls][text_column]

        for text in class_texts:
            for _ in range(3):  # Создаем 3 новых текста для каждого исходного текста
                try:
                    new_t = time()
                    new_text = paraphrase_text_rugpt3(text)
                    end_t = time()
                    print(f"Elapsed_time: {end_t - new_t}")
                    augmented_data.append({text_column: new_text, class_column: cls})
                except Exception as e:
                    print(f"Ошибка при работе с ruGPT-3: {e}")
                    continue

    augmented_df = pd.DataFrame(augmented_data)
    updated_dataset = pd.concat([dataset, augmented_df], ignore_index=True)

    print(f"Новый размер датасета: {len(updated_dataset)}")
    return updated_dataset


def paraphrase_text_rugpt3(text):
    """
    Перефразирует текст с использованием модели ruGPT-3.
    :param text: Исходный текст для перефразирования
    :param temperature: Параметр "temperature" для управления случайностью генерации
    :return: Перефразированный текст
    """

    prompt = "Перефразируй текст с те же смыслом с тем же объемомо: "
    input_ids = tokenizer.encode(prompt + text, return_tensors="pt", max_length=1024, truncation=True).to(device)

    output = model.generate(
        input_ids,
        max_length=len(input_ids[0]) + 400,
        num_return_sequences=1,
        no_repeat_ngram_size=3,
        temperature=0.75,
        top_k=50,
        top_p=0.9,
        do_sample=True,
        num_beams=5
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    paraphrased_text = generated_text.replace(prompt, "").strip()
    print(text)
    print("|||||||||||||||||||||||")
    print(paraphrased_text)
    print("_________________________________________________")

    return paraphrased_text


# df = pd.read_csv(
#     "/Users/denismazepa/Desktop/Py_projects/VKR/datasets/datasets_final/train_refactored_lematize_cut_final.csv",
#     dtype={'RGNTI1': str, 'RGNTI2': str, 'RGNTI3': str})
# print(1)
# df_7 = df[(df['RGNTI1'] == '38') & (df['RGNTI2'] == '38.61')]
# df_new = augment_dataset_with_rugpt3(df_7, 'body', 'RGNTI3')
# print(len(df_new))
# print(df_new['RGNTI3'].value_counts())
# df.to_csv("new_cut.scv")

paraphrase_text_rugpt3(
    "Проблемы развития пассажирских портов России."
    "Отмечается, что одной из форм взаимодействия государства и бизнеса выступает институт государственно-частного "
    "партнерства (ГЧП), который позволяет привлечь в экономику дополнительные ресурсы, перераспределить риски между "
    "государством и предпринимательским сектором, направить усилия предпринимателей на решение значимых для общества "
    "социально-экономических целей и задач. Так на основе ГЧП было осуществлено строительство нового пассажирского порта"
    " Санкт-Петербург в 2005-м году, что существенно повлияло на темпы социально-экономического развития города и региона."
    " На основе имеющейся нормативно-правовой базы видится необходимость выявления точек взаимодействия государства и частного "
    "бизнеса относительно развития пассажирских портов как элемента обеспечения экономической безопасности страны. "
    "Необходимо проработать институциональные основы формирования развития собственного производства, возможностей "
    "применения ГЧП. Это позволит развить не только транспортный аспект, но и решить экономические и "
    "социально-культурные проблемы приморских территорий")
