import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import os

def count_images(df, reports_path):
    df = df.copy()
    df = df[df.subset != "ground_truth"]
    df.loc[df.subset == "train", "type"] = "train good"
    df.loc[(df.subset == "test") & (df.anomaly == "good"), "type"] = "test good"
    df.loc[(df.subset == "test") & (df.anomaly != "good"), "type"] = "test defective"
    df_plt = df.groupby(["type"]).agg({"file":"count"})
    df_plt_sorted = df_plt.sort_index(ascending=False)
    print((df_plt_sorted))
    plt.figure(figsize=(7, 5), layout="constrained")
    sns.barplot(df_plt_sorted, x=df_plt_sorted.file, y=df_plt_sorted.index)
    plt.title("Image count")
    plt.xlabel("count")
    plt.ylabel("subset")
    plt.savefig(os.path.join(reports_path, "image_count.png"))
    plt.close()

def calc_statistics(df):
    df = df.copy()
    print("\n> load images and amend database with dimensions and statistics")
    for i in range(len(df)):
        if i % 50 == 0:
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
    print("\n> check image dimensions")
    widths = df.width.unique()
    heights = df.height.unique()
    print(widths, heights)
    assert len(widths) == 1 and len(heights) == 1, "all images should be the same size"
    assert widths[0] == heights[0], "images should be quadratic"
    print("Images are all quadratic and of same size.")

    print("\n> add size column and drop width and height column")
    df["width"] = df["width"].astype("int")
    df = df.rename(columns = {"width": "img_size"})
    df = df.drop(columns = ["height"])

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

    print("\n> check if images are all grayscale if grayscale")
    bw_any = df[df.subset != "ground_truth"].grayscale.any()
    bw_all = df[df.subset != "ground_truth"].grayscale.all()
    print("any:", bw_any, "all:", bw_all)
    assert (not bw_any) or bw_all, "images should be all 'grayscale' or none"
    print("If images are grayscale, all images are grayscale.")

    return df

def calc_mask_coverage(df):
    df = df.copy()
    print("> calculate anomaly coverage and add to database", "pixels")
    df_mask = df[df.subset == "ground_truth"].copy()
    df_mask["anomaly_coverage"] = df_mask.v_mean / 255
    df = pd.concat([df, df_mask["anomaly_coverage"]], axis=1)
    df.loc[(df.subset == "test") & (df.anomaly != "good"), "anomaly_coverage"] = df[(df.subset == "ground_truth")].anomaly_coverage.values
    return df

def analyze_data(df, reports_path):
    print(reports_path)
    print(df.head())

    count_images(df, reports_path)
    df = calc_statistics(df)
    df = calc_image_dimensions(df)
    df = calc_image_color(df)
    df = calc_mask_coverage(df)

    print(df.head())

    return df
