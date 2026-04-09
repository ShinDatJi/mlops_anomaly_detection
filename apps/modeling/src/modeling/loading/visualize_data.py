import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os

def plot_anomaly_coverage(df, reports_path):
    print("plot anomaly coverage")
    df1 = df[df.subset == "ground_truth"].copy()
    df1["anomaly_coverage"] = df1.v_mean / 255
    sns.barplot(df1, x = df1.anomaly_coverage, y = df1.anomaly, orient = "h", errorbar="pi")
    plt.title("Relative anomaly coverage")
    plt.xlim([0, 1])
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("anomaly coverage")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(reports_path, "anomaly_coverage.png"))
    plt.close()

def plot_hsv_distribution(df, reports_path):
    print("plot hsv distribution")
    channels = {"h": "Hue", "s": "Saturation", "v": "Value"}

    def hsv_stripplot(df, channel):
        sns.stripplot(x=df[channel + "_mean"], hue=df.anomaly, dodge = True)
        plt.xlabel("mean")
        plt.legend(bbox_to_anchor=(0, -0.1, 1, -0.1), ncols=2,  loc='upper left', borderaxespad=0, mode="expand")
        plt.title(channels[channel] + " distribution")

    df2 = df[df.subset != "ground_truth"].copy()

    # stripplot mean distribution
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    hsv_stripplot(df2, "h")
    plt.subplot(1, 3, 2)
    hsv_stripplot(df2, "s")
    plt.subplot(1, 3, 3)
    hsv_stripplot(df2, "v")
    plt.suptitle("HSV distribution strip plot ", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_path, "hsv_distribution.png"))
    plt.close()

def plot_stat_images(df, reports_path):
    def show_title(title = None, suptitle = None):
        if suptitle:
            plt.xticks([])
            plt.yticks([])
            plt.ylabel(suptitle, fontsize=14)
        else:
            plt.axis("off")
        if title != None:
            plt.title(title)

    def show_image(img, title = None, suptitle = None):
        plt.imshow(img)
        show_title(title, suptitle)

    def show_file(file, title = None, suptitle = None):
        show_image(cv2.imread(file, cv2.IMREAD_COLOR_RGB), title, suptitle)

    def calc_mean(files, img_size):
        img_mean = np.zeros((img_size, img_size, 3))
        for f in files:
            img = cv2.imread(f, cv2.IMREAD_COLOR_RGB)
            img_mean += img
        img_mean /= len(files)
        img_mean = img_mean.astype("uint8")
        return img_mean

    def calc_var(files, img_size, img_mean):
        img_var = np.zeros((img_size, img_size, 3))
        for f in files:
            img = cv2.imread(f, cv2.IMREAD_COLOR_RGB)
            img_var += (img - img_mean)**2
        img_var = img_var / len(files)

        img_max = np.ones((img_size, img_size, 3)) * 255
        img_var = np.minimum(img_var, img_max)
        img_var = img_var.astype("uint8")
        return img_var

    print("stats images")
    df3 = df.copy()
    anomalies = df3[df3.subset == "ground_truth"].anomaly.unique()
    cols = len(anomalies) + 1
    rows = 4
    img_size = df3.img_size.iloc[0]

    plt.figure(figsize=(3 * cols, 3 * rows), layout="constrained")

    # good
    plt.subplot(rows, cols, 1, frameon=False)
    file = df3[(df3.subset == "test") & (df3.anomaly == "good")].file.iloc[0]
    show_file(file, "good", "sample")

    # good mean
    plt.subplot(rows, cols, cols + 1, frameon=False)
    img_mean_good = calc_mean(df3[(df3.subset == "test") & (df3.anomaly == "good")].file, img_size)
    show_image(img_mean_good, None, "mean")

    # good var 
    plt.subplot(rows, cols, cols * 2 + 1, frameon=False)
    img_std_good = calc_var(df3[(df3.subset == "test") & (df3.anomaly == "good")].file, img_size, img_mean_good)
    show_image(img_std_good, None, "variance")

    # anomalies
    for c, anomaly in enumerate(anomalies):
        # anomaly
        plt.subplot(rows, cols, c + 2, frameon=False)
        file = df3[(df3.subset == "test") & (df3.anomaly == anomaly)].file.iloc[0]
        show_file(file, anomaly)

        # anomaly mean
        plt.subplot(rows, cols, cols + c + 2, frameon=False)
        img_mean = calc_mean(df3[(df3.subset == "test") & (df3.anomaly == anomaly)].file, img_size)
        show_image(img_mean)

        # anomaly var
        plt.subplot(rows, cols, cols * 2 + c + 2, frameon=False)
        img_std = calc_var(df3[(df3.subset == "test") & (df3.anomaly == anomaly)].file, img_size, img_mean)
        show_image(img_std)

        # ground_truth var
        plt.subplot(rows, cols, cols * 3 + c + 2, frameon=False)
        img_mean = calc_mean(df3[(df3.subset == "ground_truth") & (df3.anomaly == anomaly)].file, img_size)
        img_std = calc_var(df3[(df3.subset == "ground_truth") & (df3.anomaly == anomaly)].file, img_size, img_mean)
        if c == 0:
            show_image(img_std, None, "ground truth")
        else:
            show_image(img_std)

    plt.suptitle("Variance images\n", fontsize=16)
    plt.savefig(os.path.join(reports_path, "variance_images.png"))
    plt.close()

def visualize_data(df, reports_path):
    plot_anomaly_coverage(df, reports_path)
    plot_hsv_distribution(df, reports_path)
    plot_stat_images(df, reports_path)
