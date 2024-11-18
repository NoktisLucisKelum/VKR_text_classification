import pandas as pd
import matplotlib.pyplot as plt
from transformers import BertTokenizer


def plot_bar_chart_frequency(df, column_name: str, name_of_plot: str):
    # Подсчитываем встречаемость каждого числа
    value_counts = df[column_name].value_counts()

    # Создаем столбчатую диаграмму
    plt.figure(figsize=(10, 6))  # Устанавливаем размер графика
    plt.bar(value_counts.index.astype(str), value_counts.values)  # Преобразуем индексы в строки

    # Настраиваем оси и заголовок
    plt.xlabel('RGNTI1')
    plt.ylabel('Частота встречаемости')
    plt.title('Частота встречаемости статей с данной тематикой. ' + name_of_plot)

    # Отображаем график
    plt.show()


def analyze_tokens(df: pd.DataFrame, dataset_name: str):
    # Инициализируем токенайзер BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print(1)
    def get_tokenized_length(text):
        tokens = tokenizer.encode(text, add_special_tokens=True)
        return len(tokens)
    print(2)
    df['tokenized_length'] = df['body'].apply(get_tokenized_length)
    print(3)
    # Визуализируем распределение длины токенизированных текстов
    plt.figure(figsize=(10, 6))
    plt.hist(df['tokenized_length'], bins=50, color='blue', alpha=0.7)
    plt.title(f'Distribution of Tokenized Text Lengths in {dataset_name}')
    plt.xlabel('Tokenized Length')
    plt.ylabel('Frequency')
    plt.show()

    # Вычисляем процентиль для определения подходящего max_len
    percentile_95 = df['tokenized_length'].quantile(0.95)
    percentile_99 = df['tokenized_length'].quantile(0.99)

    print(f"95th percentile length: {percentile_95}")
    print(f"99th percentile length: {percentile_99}")


def word_counting(df: pd.DataFrame, dataset_name: str):
    def word_count(text):
        return len(text.split())

    df['word_count'] = df['body'].apply(word_count)
    average_word_count = df['word_count'].mean()
    print(f'Средняя длина текстов в словах в {dataset_name}: {average_word_count}')


df = pd.read_csv(
    '/Users/denismazepa/Desktop/Py_projects/VKR/datasets/datasets_final/test_refactored_lematize_cut_final.csv')
df_2 = pd.read_csv(
    '/Users/denismazepa/Desktop/Py_projects/VKR/datasets/datasets_final/train_refactored_lematize_cut_final.csv')
analyze_tokens(df_2, "Train dataset")
analyze_tokens(df, "Test dataset")
word_counting(df_2, "Train dataset")
word_counting(df, "Test dataset")
# plot_bar_chart_frequency(df, "RGNTI1", "Тестовый набор")
