import os
import sys
import torch
import joblib


def check_files_with_format(directory, file_format, n):
    """
    Проверяет, что в указанной директории находится n файлов с заданным форматом.

    :param directory: Путь к директории.
    :param file_format: Формат файлов (например, ".txt", ".jpg").
    :param n: Количество файлов, которое нужно проверить.
    :return: True, если файлов с указанным форматом ровно n, иначе False.
    """
    if not os.path.isdir(directory):
        raise ValueError(f"Указанный путь '{directory}' не является директорией.")

    files_with_format = [f for f in os.listdir(directory) if f.endswith(file_format)]

    return len(files_with_format) == n


def run_text_classifier(directory, model_name, input_text):
    """
    Загружает модель PyTorch для текстовой классификации из указанной директории,
    выполняет предсказание и возвращает двузначный номер из имени файла модели.

    :param directory: Путь к директории с моделями.
    :param model_name: Имя файла модели (например, "model_12.pth").
    :param input_text: Строка, которую нужно классифицировать.
    :return: Двузначный номер из имени файла модели и предсказание.
    """
    model_path = os.path.join(directory, model_name)

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Модель '{model_name}' не найдена в директории '{directory}'.")

    # Загружаем модель
    model = torch.load(model_path)
    model.eval()  # Переключаем модель в режим оценки

    # Преобразуем строку в тензор (пример, адаптируйте под вашу модель)
    # Предполагается, что модель принимает токенизированный текст в виде тензора
    input_tensor = torch.tensor([ord(c) for c in input_text]).unsqueeze(0)  # Пример токенизации

    # Выполняем предсказание
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()  # Получаем индекс предсказания

    print(f"Предсказание первой модели: {prediction}")

    # Извлекаем двузначный номер из имени файла модели
    number = ''.join(filter(str.isdigit, model_name))[-2:]  # Берем последние две цифры

    return number, prediction


def find_and_run_second_model(directory, number, input_text):
    """
    Находит модель PyTorch во второй директории по двузначному номеру,
    выполняет предсказание и возвращает результат.

    :param directory: Путь к директории с моделями.
    :param number: Двузначный номер для поиска модели.
    :param input_text: Строка, которую нужно классифицировать.
    :return: Предсказание второй модели.
    """
    for filename in os.listdir(directory):
        if filename.endswith(".pth") and number in filename:
            model_path = os.path.join(directory, filename)

            # Загружаем модель
            model = torch.load(model_path)
            model.eval()  # Переключаем модель в режим оценки

            # Преобразуем строку в тензор (пример, адаптируйте под вашу модель)
            input_tensor = torch.tensor([ord(c) for c in input_text]).unsqueeze(0)  # Пример токенизации

            # Выполняем предсказание
            with torch.no_grad():
                output = model(input_tensor)
                prediction = torch.argmax(output, dim=1).item()  # Получаем индекс предсказания

            print(f"Предсказание второй модели: {prediction}")
            return prediction

    raise FileNotFoundError(f"Модель с номером {number} не найдена в директории '{directory}'.")


def find_joblib_model_and_predict(directory, prefix, input_text):
    """
    Ищет модель в формате joblib в указанной директории,
    имя которой начинается с заданного префикса, выполняет предсказание и возвращает результат.

    :param directory: Путь к директории с моделями.
    :param prefix: Префикс для поиска файла модели.
    :param input_text: Строка, которую нужно классифицировать.
    :return: Предсказание модели.
    """
    for filename in os.listdir(directory):
        if filename.endswith(".joblib") and filename.startswith(str(prefix)):
            model_path = os.path.join(directory, filename)

            # Загружаем модель
            model = joblib.load(model_path)

            # Преобразуем строку в формат данных для модели (пример, адаптируйте под вашу модель)
            input_vector = [ord(c) for c in input_text]  # Пример преобразования строки в вектор

            # Выполняем предсказание
            prediction = model.predict([input_vector])[0]  # Предполагается, что модель поддерживает метод predict

            print(f"Предсказание третьей модели: {prediction}")
            return prediction


def main():
    print("Добро пожаловать в консольное приложение!")
    print("Инструкция:")
    print("1. Введите строку для обработки.")
    print("2. Для выхода из приложения введите 'exit'.")
    print("Примечание: Убедитесь, что ваши модели первого и второго уровня в формате .h5 находятся в папках "
          "'models' и 'models' соотвественно")

    dict_of_directories = {"Модель 1": ["./models", 1, "h5"], "Модель 2": ["./models", 64, "h5"], "Модель 3": ["./models", 300, "joblib"],}

    models_directory_lvl_1 = "./models"
    models_directory_lvl_2 = "./models"
    models_directory_lvl_3 = "./models"

    first_model_name = 'random'

    for i in dict_of_directories.keys():
        dict_name = dict_of_directories[i][0]
        dict_numb = dict_of_directories[i][1]
        dict_format = dict_of_directories[i][2]
        if not os.path.exists(dict_name):
            print(f"Папка  для {dict_name} не найдена. Создайте её и поместите туда модели формата .h5.")
            sys.exit(1)
        if not check_files_with_format(dict_name, dict_format, dict_numb):
            print(f"В папке {dict_name} нет файлов формата {dict_format} в нужном количестве")
            sys.exit(1)

    while True:
        user_input = input("\nВведите строку (или 'exit' для выхода): ")
        if user_input.lower() == "exit":
            print("До свидания!")
            break

        print("\nОбработка строки через цепочку моделей...")
        number, prediction1 = run_text_classifier(models_directory_lvl_1, first_model_name, user_input)
        prediction2 = find_and_run_second_model(models_directory_lvl_2, number, user_input)
        prediction3 = find_joblib_model_and_predict(models_directory_lvl_3, prediction2, user_input)
        print(f"Результат работы третьей модели: {prediction3}")


if __name__ == "__main__":
    main()
