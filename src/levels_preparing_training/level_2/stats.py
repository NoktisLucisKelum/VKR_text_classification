import pandas as pd

# df = pd.read_csv(
#     "/Users/denismazepa/Desktop/Py_projects/VKR/src/preprocessing/train_big_augmented_uncut_preprocessed_final_level_2_3.csv",
#     dtype={'RGNTI1': str, 'RGNTI2': str, 'RGNTI3': str})
#
# part_34 = df[df["RGNTI1"] == '34']
# part_47 = df[df["RGNTI1"] == '47']
# part_06 = df[df["RGNTI1"] == '06']
# part_31 = df[df["RGNTI1"] == '31']
# print(34, len(part_34))
# print(part_34["RGNTI2"].value_counts())
# print(47, len(part_47))
# print(part_47["RGNTI2"].value_counts())
# print('06', len(part_06))
# print(part_06["RGNTI2"].value_counts())
# print(31, len(part_31))
# print(part_31["RGNTI2"].value_counts())

df = pd.read_csv("/Users/denismazepa/Desktop/Py_projects/VKR/src/preprocessing/train_big_augmented_uncut_preprocessed_final_level_2_3.csv",
                 dtype={'RGNTI1': str, 'RGNTI2': str, 'RGNTI3': str})
print(df["RGNTI1"].value_counts())