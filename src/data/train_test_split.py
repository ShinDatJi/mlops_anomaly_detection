import pandas as pd

mad_file = "./data/mad.csv"
mad_train_file = "./data/mad_train.csv"
mad_test_file = "./data/mad_test.csv"

random_state = 42
defect_train_test_split_frac = 0.5

df = pd.read_csv(mad_file, index_col=0)
print(df.head())

df_train_good = df[df.subset == "train"].copy()
df_test_good = df[(df.subset == "test") & (df.anomaly == "good")].copy()

df_defect = df[(df.subset == "test") & (df.anomaly != "good")].copy()
df_defect["file_ground_truth"] = df_defect.file.str.replace(".png", "_mask.png").str.replace("/test/", "/ground_truth/")

df_train_defect_arr = []
df_test_defect_arr = []
for c in df_defect.category.unique():
    df_cat = df_defect[df_defect.category == c]
    for a in df_cat.anomaly.unique():
        df_sample = df_cat[df_cat.anomaly == a]
        df_sample_train = df_sample.sample(frac=defect_train_test_split_frac, random_state=random_state)
        df_sample_test = df_sample[df_sample.index.isin(df_sample_train.index) == False]
        df_train_defect_arr.append(df_sample_train)
        df_test_defect_arr.append(df_sample_test)
df_train_defect = pd.concat(df_train_defect_arr, axis=0)
df_test_defect = pd.concat(df_test_defect_arr, axis=0)

df_train = pd.concat([df_train_good, df_train_defect], axis=0)
df_train = df_train.drop("subset", axis=1)
# df_train = df_train.sample(frac=1, random_state=random_state)

df_test = pd.concat([df_test_good, df_test_defect], axis=0)
df_test = df_test.drop("subset", axis=1)
# df_test = df_test.sample(frac=1, random_state=random_state)

assert not df_train.file.isin(df_test.file).any(), "no train data should be in test database"
assert not df_test.file.isin(df_train.file).any(), "no test data should be in train database"

df_train.to_csv(mad_train_file)
df_test.to_csv(mad_test_file)
