import os
import pandas as pd

data_raw_dir = os.environ["DATA_RAW_DIR"]
data_processed_dir = os.environ["DATA_PROCESSED_DIR"]
data_raw_file = os.environ["DATA_RAW_FILE"]

os.makedirs(data_processed_dir, exist_ok=True)

# read MAD file structure into a dataframe
def create_file_db(path):
    db = []
    for root, dirs, files in os.walk(path):
        print(path)
        print(root)
        paths = root.replace(path, '').replace('\\', '/').split('/')
        if paths[0] == '':
            paths = paths[1:] # remove root path
        print(paths)
        for f in files:
            entry = []
            entry.append(paths) # array of paths (folder structure)
            entry.append(f.split('.')[0]) # filename
            entry.append(f.split('.')[1]) # file extension
            entry.append(root.replace('\\', '/') + '/' + f) # complete path of a file
            db.append(entry)
    return db

def prepare_files(df):
    # check extensions
    print("\n> list file extensions")
    extensions = df.extension.unique()
    print(extensions)
    assert "png" in extensions, "data should contain 'png' files"
    print("Image type is png.")

    # remove all non png files
    print("> remove all non png files")
    df = df[df.extension == "png"]
    df = df.drop(columns="extension")

    # check location of image files
    print("\n> list image path depths")
    df["path_depth"] = df.path.apply(lambda p: len(p))
    img_depths = df.path_depth.unique()
    print(img_depths)
    assert img_depths == [2], "images should only be in depth 2"
    print("Images have all path depth of 2.")

    print("> expand paths and remove path column")
    df["path1"] = df.path.apply(lambda p: p[0])
    df["path2"] = df.path.apply(lambda p: p[1])
    df = df.drop(columns=["path", "path_depth"])
    
    print("\nDatabase cleaned, only containing images.")
    return df

def prepare_folders(df):
    print("\n> list unique values of path1")
    path1 = df.path1.unique()
    print(path1)
    assert (path1 == ["ground_truth", "test", "train"]).all(), "all image subsets should be available"
    print("This is the subset of images used for training and evaluation.")

    print("\n> list unique values of path2")
    path2 = df.path2.unique()
    print(path2)
    assert "good" in path2, "'good' anomaly should be available"
    print("This is the anomaly type of images. There is also a 'good' anomaly type.")

    print("\n> Rename the columns accordingly")
    df = df.rename(columns={"path1": "subset", "path2": "anomaly"})
    return df

def check_file_names(df):
    print("\n> split file names by letter")
    splitted = df.name.str.split("").apply(lambda s: s[1:-1])
    print("> count characters")
    characters = splitted.apply(lambda s: len(s)).unique()
    print(characters)
    assert 8 in characters, "there should be 8 character file names"
    assert 3 in characters, "there should be 3 character file names"
    assert len(characters == 2), "there should only be 3 and 8 character file names"
    print("There are only file names with 3 and 8 characters.")

    print("\n> list first 3 characters")
    chars1 = splitted.apply(lambda s: s[0]).unique()
    chars2 = splitted.apply(lambda s: s[1]).unique()
    chars3 = splitted.apply(lambda s: s[2]).unique()
    print(chars1)
    print(chars2)
    print(chars3)
    assert all([int(c) in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] for c in chars1]), "first character should be a number"
    assert all([int(c) in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] for c in chars2]), "second character should be a number"
    assert all([int(c) in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] for c in chars3]), "third character should be a number"
    print("First 3 character are a numbering of the images.")

    print("\n> list last 5 characters of 8 character names")
    df8_char = df[splitted.apply(lambda s: len(s) == 8)]
    df8_chars = df8_char.name.apply(lambda s: s[3:]).unique()
    print(df8_chars)
    assert df8_chars == ['_mask'], "files with 8 characters should end with '_mask'" 
    print("All images with 8 character names have '_mask' as extension.")

    print("\n> check '_mask' images subset")
    df8_chars_subsets = df8_char.subset.unique()
    print(df8_chars_subsets)
    assert df8_chars_subsets == ['ground_truth'], "'_mask' images should be in subset 'ground_truth'"
    print("All '_mask' images are in the subset 'ground_truth'.")

    print("\n> check if all ground_truth images have '_mask' extension.")
    df8_char_length = len(df8_char)
    df_ground_truth_length = len(df[df.subset=="ground_truth"])
    print("'_mask' count:", df8_char_length, "'ground_truth' count:", df_ground_truth_length)
    assert df8_char_length == df_ground_truth_length, "all 'ground_truth' images should end with '_mask'"
    print("'ground_truth' images are extended with '_mask' string")

def check_consistency(df):
    print("\n> list subsets")
    subsets = df.subset.unique()
    print(subsets)
    assert "train" in subsets and "test" in subsets and "ground_truth" in subsets, "subsets should contain 'ground_truth', 'test' and 'train'"
    assert len(subsets) == 3, "there should only be 3 subsets"
    print("'train', 'test' and 'ground_truth' subsets are present.")

    print("\n> group by subset")
    group_subset = df.groupby("subset").anomaly.agg(lambda c: c.unique()).to_frame()
    print(group_subset)
    all_train_good = all(group_subset.loc["train"].anomaly == 'good')
    assert all_train_good, "train subset should contain only 'good' anomaly"
    print("'train' subset has only 'good' images.")

    print("\n> check occurrence of 'good' images in subsets")
    good_subsets = df[df.anomaly=="good"].subset.unique()
    print(good_subsets)
    assert "test" in good_subsets and "train" in good_subsets and len(good_subsets == 2), "all 'good' images should be only in 'test' and 'train' subsets"
    print("'good' images are only in test and train subsets for each category.")

    print("\n> check if anomaly folders and file names in 'test' and 'ground_truth' subsets are the same")
    df_test = df[(df.anomaly!="good") & (df.subset=="test")][["anomaly", "name"]].reset_index(drop=True)
    df_ground_truth = df[df.subset=="ground_truth"][["anomaly", "name"]].reset_index(drop=True)
    df_ground_truth["name"] = df_ground_truth.name.apply(lambda s: s[:3]) # take only numbering of file name into account
    print((df_test == df_ground_truth).all())
    assert (df_test == df_ground_truth).all().all(), "anomaly folders and file names should be the same for 'test' and 'ground_truth'"
    print("Anomaly folders and file names in 'test' and 'ground_truth' are the same.")

def save_database(df):
    print("Images are well structured in the data set.")
    print("> remove 'name' columns, reorder columns and save database to disk")
    df = df.drop(columns=["name"]).reset_index(drop=True)
    df = df.loc[:, ["subset", "anomaly", "file"]]
    df.to_csv(data_raw_file)
    print(df.head())
    return df

print("> create file database reading the MVTec AD folder structure")

db = create_file_db(data_raw_dir)
df = pd.DataFrame(db, columns=["path", "name", "extension", "file"])
print(df.head())

df = prepare_files(df)
print(df.head())

df = prepare_folders(df)
print(df.head())

check_file_names(df)

check_consistency(df)

df = save_database(df)

print("\n> summary")
print(df.groupby(["subset", "anomaly"]).agg("count").sort_values(by="subset", ascending=False))
