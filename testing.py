import pandas as pd

# df = pd.read_csv("datatsets_from_git/test_ru.csv", on_bad_lines='skip', delimiter=' ')
# # print(df['id\ttitle\tbody\tkeywords\tcorrect\tRGNTI1\tRGNTI2\tRGNTI3'].head())
# print(df.columns.tolist())
# print(df.head())

df = pd.read_csv("datasets/datasets_final/train_refactored.csv")
print(df.columns.tolist())
