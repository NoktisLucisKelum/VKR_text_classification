import json


def load_json(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def save_json(data: json, file_path: str):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def create_nested_dict(json1: json, json2: json, json3: json):
    nested_dict = {}

    for key1, value1 in json1.items():
        nested_dict[key1] = {}

        for key2, value2 in json2.items():
            if key2.startswith(key1):
                nested_dict[key1][key2] = {}

                for key3, value3 in json3.items():
                    if key3.startswith(key2):
                        nested_dict[key1][key2][key3] = value3

    return nested_dict


def main():
    # Загрузка данных из файлов
    json1 = load_json('../grnti/GRNTI_1_ru.json')
    json2 = load_json('../grnti/GRNTI_2_ru.json')
    json3 = load_json('../grnti/GRNTI_3_ru.json')

    # Создание трехуровневого словаря
    nested_dict = create_nested_dict(json1, json2, json3)

    # Сохранение результата в новый JSON файл
    save_json(nested_dict, '../result.json')


if __name__ == '__main__':
    main()