from dataset_prepare_for_training import *


df = pd.read_csv("/Users/denismazepa/Desktop/Py_projects/VKR/datasets/datatsets_from_git/train/train_ru_work.csv",
                 sep="\t", on_bad_lines='skip')

print(df.columns)
# print(df.columns)
preprocessor = TextPreprocessor(df)

# Последовательно применяем функции
preprocessor.drop_nan()
preprocessor.chem_formula_prepare()
preprocessor.phys_formula_prepare()
preprocessor.remove_english_strings()
preprocessor.lemmatize(['title', 'body', 'keywords'])
preprocessor.remove_punctuation(['title', 'body'])
# # preprocessor.stem(['text_column'])
preprocessor.remove_stop_words(['title', 'body'])
# preprocessor.convert_to_word_list(['title', 'body', 'keywords'])
preprocessor.remove_second_index(["RGNTI1", "RGNTI2", "RGNTI3"])
preprocessor.drop_columns(["correct"])
preprocessor.merge_and_drop_columns('body', 'title')
# preprocessor.printing('body')
# preprocessor.printing('keywords')
preprocessor.repare_columns()

preprocessor.save_to_csv("../VKR/datasets/datasets_final/train_refactored_lematize_lulz.csv")
