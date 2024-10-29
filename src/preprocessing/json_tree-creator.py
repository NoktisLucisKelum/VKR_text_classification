import json
import pandas as pd


class JSONProcessor:
    def __init__(self, file_path1: str, file_path2: str, file_path3: str, result_file_path: str):
        self.file_path1 = file_path1
        self.file_path2 = file_path2
        self.file_path3 = file_path3
        self.result_file_path = result_file_path

    def load_json(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def save_json(self, data: dict, file_path: str):
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

    def create_nested_dict(self, json1: dict, json2: dict, json3: dict):
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

    def process(self):
        # Загрузка данных из файлов
        json1 = self.load_json(self.file_path1)
        json2 = self.load_json(self.file_path2)
        json3 = self.load_json(self.file_path3)

        # Создание трехуровневого словаря
        nested_dict = self.create_nested_dict(json1, json2, json3)

        # Сохранение результата в новый JSON файл
        self.save_json(nested_dict, self.result_file_path)


class DataFrameToJsonTree:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def create_tree(self):
        tree = {}

        for _, row in self.dataframe.iterrows():
            level1, level2, level3 = row["RGNTI1"], row["RGNTI2"], row["RGNTI3"]

            if level1 not in tree:
                tree[level1] = {}

            if level2 not in tree[level1]:
                tree[level1][level2] = []
            if level3 not in tree[level1][level2]: tree[level1][level2].append(level3)

        return tree

    def save_to_json(self, filename):
        tree = self.create_tree()
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(tree, file, ensure_ascii=False, indent=4)



# processor = JSONProcessor('../../grnti/GRNTI_1_ru.json', '../../grnti/GRNTI_2_ru.json', '../../grnti/GRNTI_3_ru.json', '../../result.json')
# processor.process()

df = pd.read_csv("/Users/denismazepa/Desktop/Py_projects/VKR/datasets/datasets_final/test_refactored_"
                 "lematize_cut_final.csv")
preproc = DataFrameToJsonTree(df)
preproc.create_tree()
preproc.save_to_json("result_from_test_datatset.json")
