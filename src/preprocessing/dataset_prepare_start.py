from dataset_prepare_for_training import *


df = pd.read_csv("/Users/denismazepa/Desktop/Py_projects/VKR/datasets/datatsets_from_git/test/test_ru_work.csv",
                 sep="\t", on_bad_lines='skip')

print(df.columns)
# print(df.columns)
preprocessor = TextPreprocessor(df)

# Последовательно применяем функции
preprocessor.drop_nan()
print(1)
preprocessor.chem_formula_prepare()
preprocessor.phys_formula_prepare()
print(2)
preprocessor.remove_english_strings()
print(3)
preprocessor.remove_numbers()
print(3.5)
preprocessor.lemmatize(['title', 'body', 'keywords'])
print(4)
preprocessor.remove_punctuation(['title', 'body'])
print(5)
# # preprocessor.stem(['text_column'])
preprocessor.remove_stop_words(['title', 'body'])
print(6)
# preprocessor.convert_to_word_list(['title', 'body', 'keywords'])
preprocessor.split_column_value("RGNTI1", "RGNTI1_2")
print(7)
preprocessor.remove_second_index(["RGNTI1", "RGNTI2", "RGNTI3"])
print(8)
preprocessor.drop_columns(["correct"])
print(9)
preprocessor.merge_and_drop_columns('body', 'title')
print(10)
# preprocessor.printing('body')
# preprocessor.printing('keywords')
preprocessor.repare_columns()


preprocessor.save_to_csv("test_refactored_lematize_no_numbers.csv")
