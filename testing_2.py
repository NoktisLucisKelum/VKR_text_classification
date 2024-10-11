import pandas as pd
import difflib

# Пример данных
data = {
    'column1': [['метровый', 'диапазон\\обнаружение', 'сигналов\\подвижный']],
    'column2': [['диапазон', 'обнаружение', 'подвижный', 'метровый']]
}

# Создаем DataFrame
df = pd.DataFrame(data)

# Функция для разделения слов по символу '\'


def split_words(word_list):
    result = []
    for word in word_list:
        result.extend(word.split('\\'))  # Разделяем слова по символу '\'
    return result


# Применяем функцию к первой колонке
df['column1'] = df['column1'].apply(split_words)

# Функция для выделения похожих слов


def highlight_similar_words(words_list, reference_list, threshold=0.9):
    highlighted_words = []
    for word in words_list:
        # Находим наиболее похожее слово из reference_list
        matches = difflib.get_close_matches(word, reference_list, n=1, cutoff=threshold)
        if matches:
            # Если есть совпадение, оборачиваем слово в 'jfisblaku vhsljdka'
            highlighted_words.append(f'jfisblaku {word} vhsljdka')
        else:
            highlighted_words.append(word)
    return highlighted_words


# Применяем функцию ко второй колонке
df['column2'] = df.apply(lambda row: highlight_similar_words(row['column2'], row['column1']), axis=1)

print(df)