import pandas as pd
import matplotlib.pyplot as plt


def plot_bar_chart_frequency(df, column_name: str, name_of_plot: str):
    # Подсчитываем встречаемость каждого числа
    value_counts = df[column_name].value_counts()

    # Создаем столбчатую диаграмму
    plt.figure(figsize=(10, 6)) # Устанавливаем размер графика
    plt.bar(value_counts.index.astype(str), value_counts.values) # Преобразуем индексы в строки

    # Настраиваем оси и заголовок
    plt.xlabel('RGNTI1')
    plt.ylabel('Частота встречаемости')
    plt.title('Частота встречаемости статей с данной тематикой. ' + name_of_plot)

    # Отображаем график
    plt.show()


df = pd.read_csv('/Users/denismazepa/Desktop/Py_projects/VKR/datasets/datasets_final/test_refactored_lematize_cut_final.csv')
plot_bar_chart_frequency(df, "RGNTI1", "Тестовый набор")

