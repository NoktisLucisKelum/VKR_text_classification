import pandas as pd

# df = pd.read_csv("train_colide.csv", dtype={'RGNTI1': str, 'RGNTI2': str, 'RGNTI3': str})
# print(df["RGNTI1"].unique().tolist())
# print(len(df["RGNTI1"].unique().tolist()))
# print(df["RGNTI1"].value_counts())
# df_1 = pd.read_csv("train_work_big.csv", sep="\t", on_bad_lines='skip')
# print(len(df_1))

# df = pd.read_csv("/Users/denismazepa/Desktop/Py_projects/VKR/src/preprocessing/train_big_augmented_uncut_preprocessed_final.csv", dtype={'RGNTI1': str, 'RGNTI2': str, 'RGNTI3': str})
# print(df["RGNTI1"].value_counts())
# print(len(df))

df = pd.read_csv("train_big_augmented_uncut_final.csv", dtype={'RGNTI1': str, 'RGNTI2': str, 'RGNTI3': str})
print(len(df))


print("_____________________________________________")
dfx = pd.read_csv("test_work_small.csv", sep="\t", on_bad_lines='skip')
print(len(dfx))
print("_____________________________________________")
df1 = pd.read_csv("/Users/denismazepa/Desktop/Py_projects/VKR/datasets/datatsets_from_git/train/train_ru_work.csv", sep="\t", on_bad_lines='skip')
df2 = pd.read_csv("/Users/denismazepa/Desktop/Py_projects/VKR/datasets/datatsets_from_git/test/test_ru_work.csv", sep="\t", on_bad_lines='skip')
print(len(df1), len(df2))