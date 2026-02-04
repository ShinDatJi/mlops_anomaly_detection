import pandas as pd
import json
import os

config_file = os.environ["CONFIG_FILE"]
data_clean_file = os.environ["DATA_CLEAN_FILE"]
data_train_file = os.environ["DATA_TRAIN_FILE"]
data_test_file = os.environ["DATA_TEST_FILE"]

with open(config_file, "r") as f:
    config = json.load(f)

random_state = config["random_state"]
defect_train_test_split_frac = config["train_test_split"]

df = pd.read_csv(data_clean_file, index_col=0)
print(df.head())

df_train_good = df[df.subset == "train"].copy()
df_test_good = df[(df.subset == "test") & (df.anomaly == "good")].copy()

df_defect = df[(df.subset == "test") & (df.anomaly != "good")].copy()
df_defect["file_ground_truth"] = df_defect.file.str.replace(".png", "_mask.png").str.replace("/test/", "/ground_truth/")

df_train_defect_arr = []
df_test_defect_arr = []
for a in df_defect.anomaly.unique():
    df_sample = df_defect[df_defect.anomaly == a]
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

df_train.to_csv(data_train_file)
df_test.to_csv(data_test_file)
