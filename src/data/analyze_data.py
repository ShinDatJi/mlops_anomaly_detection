import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

mad_raw_file = "./data/mad_raw.csv"
mad_file = "./data/mad.csv"
mad_stats_file = "./data/mad_stats.csv"
reports_data_dir = "./reports/data/"

def count_images(df):
    df = df.copy()
    df = df[df.subset != "ground_truth"]
    df.loc[df.subset == "train", "type"] = "train good"
    df.loc[(df.subset == "test") & (df.anomaly == "good"), "type"] = "test good"
    df.loc[(df.subset == "test") & (df.anomaly != "good"), "type"] = "test defective"
    df_plt = df.groupby(["category", "type"]).agg({"file":"count"}).unstack()
    df_plt.columns = df_plt.columns.droplevel()
    df_plt = df_plt.loc[:, ["train good", "test good", "test defective"]]
    df_plt_sorted = df_plt.sort_values("train good")
    print((df_plt_sorted))
    ax = df_plt_sorted.plot(kind="barh", stacked=True, figsize=(10,8), layout="constrained", color=["tab:blue", "tab:green", "tab:orange"], width=0.9)
    ax.bar_label(ax.containers[0], label_type="center")
    ax.bar_label(ax.containers[1], label_type="center", labels=df_plt_sorted["test good"])
    ax.bar_label(ax.containers[2], label_type="center", labels=df_plt_sorted["test defective"])
    plt.title("Image count")
    plt.xlabel("count")
    plt.savefig(reports_data_dir + "image_count.png")
    plt.close()

def count_anomalies(df):
    df = df.copy()
    df = df[df.subset == "ground_truth"].groupby(["category"]).agg({"anomaly": "nunique"})
    print(df)
    sns.barplot(data=df.sort_values("anomaly", ascending=False), y="category", x = "anomaly", width=0.9)
    plt.title("Anomaly type count")
    plt.xlabel("count")
    plt.savefig(reports_data_dir + "anomaly_type_count.png")
    plt.close()

def calc_statistics(df):
    df = df.copy()
    print("\n> load images and amend database with dimensions and statistics")
    for i in range(len(df)):
        if i % 500 == 0:
            print("loaded:", i, "/", len(df))
        im = cv2.imread(df.file[i])
        img = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        df.loc[i, "width"] = img.shape[0]
        df.loc[i, "height"] = img.shape[1]
        df.loc[i, "h_mean"] = img[:, :, 0].mean()
        df.loc[i, "h_std"] = img[:, :, 0].std()
        df.loc[i, "h_min"] = img[:, :, 0].min()
        df.loc[i, "h_max"] = img[:, :, 0].max()
        df.loc[i, "s_mean"] = img[:, :, 1].mean()
        df.loc[i, "s_std"] = img[:, :, 1].std()
        df.loc[i, "s_min"] = img[:, :, 1].min()
        df.loc[i, "s_max"] = img[:, :, 1].max()
        df.loc[i, "v_mean"] = img[:, :, 2].mean()
        df.loc[i, "v_std"] = img[:, :, 2].std()
        df.loc[i, "v_min"] = img[:, :, 2].min()
        df.loc[i, "v_max"] = img[:, :, 2].max()
    return df

def calc_image_dimensions(df):
    df = df.copy()
    print("\n> check image dimensions per category")
    cat_size = df.groupby("category").agg({"width": "unique", "height": "unique"})
    cat_size["quadratic"] = cat_size.width == cat_size.height
    print(cat_size)
    assert cat_size["quadratic"].all(), "all images should be quadratic"
    print("Images are all quadratic but have different dimensions per category.")

    print("\n> add size column and drop width and height column")
    df["width"] = df["width"].astype("int")
    df = df.rename(columns = {"width": "img_size"})
    df = df.drop(columns = ["height"])

    ax = sns.barplot(y = df.category, x = df.img_size, width=0.9)
    ax.bar_label(ax.containers[0], label_type="edge", padding=-25)
    plt.title("Image size")
    plt.xlabel("size")
    plt.savefig(reports_data_dir + "image_size.png")
    plt.close()

    return df

def calc_image_color(df):
    df = df.copy()
    print("\n> find gray scaled images and add grayscale column to database")
    df["grayscale"] = df.s_mean == 0
    print("> check grayscale per subset")
    group_subset_grayscale = df.groupby(["subset"]).grayscale.all()
    print(group_subset_grayscale)
    assert group_subset_grayscale.loc["ground_truth"], "all 'ground_truth' images should be grayscale"
    print("All ground_truth images are grayscale.")

    print("\n> check if images are grayscale per category")
    df_cat = df[df.subset != "ground_truth"].groupby("category").grayscale.all()
    print(df_cat[df_cat])
    assert df_cat[df_cat].all(), "images should be all 'grayscale' in a category"
    print("'grid', 'screw' and 'zipper' images are grayscale.")

    print("\n> check if images of other categories are not grayscale")
    df_cat = df[df.subset != "ground_truth"].groupby("category").grayscale.any() == False
    print(df_cat[df_cat])
    assert df_cat[df_cat].all(), "images should be all 'color' in a category"
    print("Images in other categories are all color images.")

    return df

def calc_mask_coverage(df):
    df = df.copy()
    df_mask = df[df.subset == "ground_truth"].copy()
    df_mask["anomaly_coverage"] = df_mask.v_mean / 255

    min_pixels = 100
    print("> calculate the minimal image size if the anomalies should still cover about", min_pixels, "pixels")
    df_mask["min_img_size"] = np.sqrt(min_pixels / df_mask.anomaly_coverage).astype(int)
    df_plot = df_mask.groupby("category").agg({
        "img_size": "median",
        "anomaly_coverage": "min", 
        "min_img_size": "max"
        }).rename({"anomaly_coverage": "min_anomaly_coverage"}, axis=1)
    print(df_plot)

    print("> add new columns to database")
    df = pd.concat([df, df_mask[["anomaly_coverage", "min_img_size"]]], axis=1)
    df.loc[(df.subset == "test") & (df.anomaly != "good"), "anomaly_coverage"] = df[(df.subset == "ground_truth")].anomaly_coverage.values
    df.loc[(df.subset == "test") & (df.anomaly != "good"), "min_img_size"] = df[(df.subset == "ground_truth")].min_img_size.values

    ax = sns.barplot(x=df_plot.img_size, y=df_plot.index, label="original size", width=0.9)
    ax.bar_label(ax.containers[0], label_type="edge", padding=-25)
    ax = sns.barplot(x=df_plot.min_img_size, y=df_plot.index, label="reduced size", width=0.9)
    ax.bar_label(ax.containers[1], label_type="edge", padding=-25)
    plt.title("Minimal image size, so that anomalies cover at least " + str(min_pixels) + " pixels of image")
    plt.xlabel("size")
    # plt.legend(bbox_to_anchor=(0, -0.1, 1, -0.1),  loc='upper left', borderaxespad=0, mode="expand", ncols=2)
    plt.savefig(reports_data_dir + "min_image_size.png")
    plt.close()

    return df

def save_database(df):
    df.to_csv(mad_stats_file)
    df = df.copy()
    print("> save database for modeling")
    df = df[["category", "subset", "anomaly", "img_size", "grayscale", "anomaly_coverage", "min_img_size", "file"]]
    df.to_csv(mad_file)
    return df

df = pd.read_csv(mad_raw_file)
print(df.head())

# df = df[(df.category == "bottle") | (df.category == "zipper")].reset_index(drop=True)

count_images(df)
count_anomalies(df)
df = calc_statistics(df)
df = calc_image_dimensions(df)
df = calc_image_color(df)
df = calc_mask_coverage(df)
df = save_database(df)

print(df.head())
