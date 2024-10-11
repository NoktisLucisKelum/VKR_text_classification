import pandas as pd
import matplotlib.pyplot as plt


def plot_bar_chart_frequency(df, column_name):
    # Подсчитываем встречаемость каждого числа
    value_counts = df[column_name].value_counts()

    # Создаем столбчатую диаграмму
    plt.figure(figsize=(10, 6)) # Устанавливаем размер графика
    plt.bar(value_counts.index.astype(str), value_counts.values) # Преобразуем индексы в строки

    # Настраиваем оси и заголовок
    plt.xlabel('Двузначные числа')
    plt.ylabel('Частота встречаемости')
    plt.title('Частота встречаемости двузначных чисел')

    # Отображаем график
    plt.show()


df = pd.read_csv('datasets/datasets_final/train_refactored_lematize.csv')
plot_bar_chart_frequency(df, "RGNTI1")




