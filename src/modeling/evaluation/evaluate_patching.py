import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import evaluate_simple

def plot_false_predictions(df_predict, df_predict_patch, patches_x, patches_y, threshold, report_path=None):
    patched_pred_proba = df_predict_patch.pred_proba.values.reshape((-1, patches_x * patches_y))
    plt.figure(figsize=(10,4))
    plt.subplot(1, 2, 1)
    fn = 0
    for i in range(len(df_predict)):
        if (df_predict.real[i] == 1) & (df_predict.pred[i] == 0):
            plt.plot(patched_pred_proba[i], alpha=0.5)
            fn += 1
    plt.xlim(0, patches_x * patches_y - 1)
    plt.ylim(0, threshold)
    plt.xlabel("Patch")
    plt.ylabel("Probability")
    plt.title(f"False Negatives: {fn}")
    plt.grid(alpha = 0.25, axis="x")
    # plt.legend()
    plt.subplot(1, 2, 2)
    fp = 0
    for i in range(len(df_predict)):
        if (df_predict.real[i] == 0) & (df_predict.pred[i] == 1):
            plt.plot(patched_pred_proba[i], alpha=0.5)
            fp += 1
    plt.xlim(0, patches_x * patches_y - 1)
    plt.ylim(0, 1)
    plt.axhline(threshold, color="red", alpha=0.5)
    plt.text(0.02 * patches_x * patches_y, threshold + 0.01, f"threshold: {threshold:0.2f}", verticalalignment="bottom")
    plt.xlabel("Patch")
    plt.ylabel("Probability")
    plt.title(f"False Positives: {fp}")
    plt.grid(alpha = 0.25, axis="x")

    plt.tight_layout()
    if report_path:
        plt.savefig(os.path.join(report_path, "false_predictions.png"))
    plt.close()

def plot_mean_probabilites(df_predict_patch, one_line, patches, patches_x, patches_y, threshold, report_path=None):

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), width_ratios=[2, 1])
    fig = plt.figure(figsize=(10, 6))
    subfigs = fig.subfigures(1, 2, width_ratios=[2, 1])

    patched_pred_proba = df_predict_patch.pred_proba.values.reshape((-1, patches_x * patches_y))
    proba_0_mean = patched_pred_proba[:one_line].mean(axis=0)
    ax1 = subfigs[0].subplots(1, 1)
    ax1.plot(proba_0_mean, label="class: negative")
    ax1.axhline(proba_0_mean.max(), alpha=0.5, color="gray")
    proba_1_mean = patched_pred_proba[one_line:].mean(axis=0)
    ax1.plot(proba_1_mean, label="class: positive")
    # proba_0_max = patch_pred_proba[:one_line_unpatched].max(axis=0)
    # plt.plot(proba_0_max, color="tab:blue", alpha=0.5)
    ax1.axhline(proba_1_mean.min(), alpha=0.5, color="gray")
    ax1.fill_between(range(patches_x * patches_y), proba_0_mean.max(), proba_1_mean.min(), alpha=0.25, color="gray")
    # plt.text(0.02 * patches**2, proba_1_mean.min() + 0.01, "Min mean positive probability", verticalalignment="bottom")
    # plt.text(0.02 * patches**2, proba_0_mean.max() - 0.01, "Max mean negative probability", verticalalignment="top")
    plt.text(0.02 * patches_x * patches_y, proba_1_mean.min() - 0.01, "Max/Min range", verticalalignment="top")
    ax1.axhline(threshold, color="red", alpha=0.5)
    ax1.text(0.02 * patches_x * patches_y, 0.01 + threshold, f"threshold: {threshold:0.2f}", verticalalignment="bottom")
    ax1.set_xlim(0, patches_x * patches_y - 1)
    ax1.set_ylim(0, 1)
    ax1.set_title("Mean prediction probabilities")
    ax1.set_xlabel("Patch")
    ax1.set_ylabel("Probability")
    ax1.grid(alpha = 0.25, axis="x")
    ax1.legend()

    # fig, (ax21, ax22) = plt.subplots(2, 1)
    axs2 = subfigs[1].subplots(2, 1)
    ax21 = axs2[0]
    ax22 = axs2[1]
    patched_pred_proba = df_predict_patch.pred_proba.values.reshape((-1, patches_y, patches_x))
    # plt.figure(figsize=(10,4))
    # plt.subplot(1,2,1)
    sns.heatmap(patched_pred_proba[:one_line].mean(axis=0), vmin=0, vmax=1, annot=True, square=True, cbar=False, ax=ax21, fmt="0.1f")
    ax21.set_xlabel(f"threshold: {threshold:0.2f}")
    ax21.set_title("Negative class mean probabilities")
    ax21.set_xticks([])
    ax21.set_yticks([])
    # plt.subplot(1,2,2)
    sns.heatmap(patched_pred_proba[one_line:].mean(axis=0), vmin=0, vmax=1, annot=True, square=True, cbar=False, ax=ax22, fmt="0.1f")
    ax22.set_xlabel(f"threshold: {threshold:0.2f}")
    ax22.set_title("Positive class mean probabilities")
    ax22.set_xticks([], labels=[])
    ax22.set_yticks([], labels=[])

    plt.tight_layout()
    if report_path:
        plt.savefig(os.path.join(report_path, "mean_probabilities.png"), bbox_inches="tight")
    plt.close()

def plot_probabilities(df_predict_patch, one_line_patch, threshold, report_path=None):
    df_sort = df_predict_patch.sort_values(by=["real", "pred_proba"])
    test_pred_proba_sorted = df_sort["pred_proba"].values

    plt.bar(range(len(test_pred_proba_sorted)), test_pred_proba_sorted, width=1)
    plt.axvline(one_line_patch, color="red", alpha=0.5)
    plt.text(one_line_patch - 0.01 * len(test_pred_proba_sorted), 0.98, "Neg/Pos", horizontalalignment="right", verticalalignment="top")
    plt.axhline(threshold, color="red", alpha=0.5)
    plt.text(0.02 * len(test_pred_proba_sorted), 0.01 + threshold, f"threshold: {threshold:0.2f}", verticalalignment="bottom")
    plt.xlim(0, len(test_pred_proba_sorted) - 1)
    plt.ylim(0, 1)
    plt.xlabel("Image")
    plt.ylabel("Probability")
    plt.title("Sorted prediction probabilities")
    # plt.legend()
    if report_path:
        plt.savefig(os.path.join(report_path, "prediction_probabilities.png"))
    plt.close()

def plot_patching(df_test, df_predict, df_predict_patch, test_images, patches, patches_x, patches_y, threshold, report_path=None):

    def plot_patches(row, col, img, i, title):
        plt.subplot(6, 4, (row - 1) * 12 + col)
        img_ori = cv2.imread(df_test.iloc[i].file)
        img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
        plt.imshow(img_ori)
        plt.axis("off")
        # plt.title(df_test.index[i])
        plt.text(0, img_ori.shape[0] * 1.075, df_test.iloc[i].file, fontsize="xx-small")
        plt.title(title)

        pred_proba = df_predict_patch.pred_proba.values[i*(patches_x*patches_y):i*(patches_x*patches_y) + patches_x*patches_y]

        plt.subplot(6, 4, (row - 1) * 12 + col + 4)
        plt.imshow(np.array(img).astype("uint8"))
        plt.hlines(range(img.shape[0] // patches_y, img.shape[0], img.shape[0] // patches_y), xmin=0, xmax=img.shape[1] - 1, color="blue", linestyle="-", linewidth=1, alpha=0.5)
        plt.vlines(range(img.shape[1] // patches_x, img.shape[1], img.shape[1] // patches_x), ymin=0, ymax=img.shape[0] - 1, color="blue", linestyle="-", linewidth=1, alpha=0.5)
        for p in range(patches_x*patches_y):
        # for r in range(patches_y):
            # for c in range(patches_x):
                if pred_proba[p] >= threshold:
                    x_offset = (img.shape[1] // patches_x)
                    y_offset = (img.shape[0] // patches_y)
                    x = (p % patches_x) * x_offset
                    y = (p // patches_x) * y_offset
                    xs = [x, x + x_offset - 1, x + x_offset - 1, x, x]
                    ys = [y, y, y + y_offset - 1, y + y_offset - 1, y]
                    plt.fill(xs, ys, color="red", alpha=0.25)
                    plt.plot(xs, ys, color="red", linewidth=1, linestyle="-", alpha=0.5)
        plt.axis("off")
        plt.title("Patch activation map", fontsize="medium")

        plt.subplot(6, 4, (row - 1) * 12 + col + 8)
        sns.heatmap(pred_proba.reshape((patches_y, patches_x)), annot=True, square=True, vmin=0, vmax=1, cbar=False, fmt="0.1f")
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(f"threshold: {threshold:0.2f}")
        plt.title("Patch probability map", fontsize="medium")

    plt.figure(figsize=(12, 18))
    tp_count = 0
    fp_count = 0
    tn_count = 0
    fn_count = 0
    # for i, img in enumerate(test_images):
    for i in np.random.permutation(len(test_images)):
        img = test_images[i]
        if (tp_count < 2) & (df_predict.real[i] == 1) & (df_predict.pred[i] == 1):
            tp_count += 1
            plot_patches(1, tp_count, img, i, "TP")
        if (fn_count < 2) & (df_predict.real[i] == 1) & (df_predict.pred[i] == 0):
            fn_count += 1
            plot_patches(1, 2 + fn_count, img, i, "FN")
        if (tn_count < 2) & (df_predict.real[i] == 0) & (df_predict.pred[i] == 0):
            tn_count += 1
            plot_patches(2, tn_count, img, i, "TN")
        if (fp_count < 2) & (df_predict.real[i] == 0) & (df_predict.pred[i] == 1):
            fp_count += 1
            plot_patches(2, 2 + fp_count, img, i, "FP")
    plt.tight_layout()
    if report_path:
        plt.savefig(os.path.join(report_path, "patching.png"))
    plt.close()

def predict_patched(df_test, df_predict_patch, patch_threshold, threshold, img_size, patches, patches_x, patches_y, patch_images):
    test_images = []
    test_pred = []
    test_real = df_test.anomaly.apply(lambda v: 0 if v == "good" else 1).values

    one_line = np.where(np.array(test_real) == 1)[0][0]

    # step = patches ** 2
    step = patches_x * patches_y
    for i in range(len(df_test)):
        img = np.array(patch_images[i*step:i*step + step])
        img = img.reshape((patches_y, patches_x, img_size, img_size, -1))
        image = np.zeros((patches_y*img_size, patches_x*img_size, 3))
        for r in range(patches_y):
            for c in range(patches_x):
                image[r*img_size:r*img_size+img_size, c*img_size:c*img_size+img_size] = img[r, c, :, :, :]
        test_images.append(image)
        if patch_threshold == "mean":
            test_pred.append((df_predict_patch.pred_proba.values[i*step:i*step+step].mean() >= threshold).astype(int))
        elif patch_threshold == "max":
            test_pred.append((df_predict_patch.pred_proba.values[i*step:i*step+step].max() >= threshold).astype(int))
        else:
            test_pred.append((df_predict_patch.pred.values[i*step:i*step+step].sum() >= patch_threshold).astype(int))

    df_predict = pd.DataFrame(test_real, columns=["real"])
    df_predict["pred"] = test_pred

    return df_predict, one_line, test_images

def test_model(data_dir, model, img_size, batch_size, threshold, df_test, patches, patches_x, patches_y, patch_threshold, report_path=None, grayscale=False):
    ds_test, patch_images, test_real = evaluate_simple.load_data(data_dir, img_size, batch_size, grayscale)
    df_predict_patch, one_line_patch, threshold = evaluate_simple.predict(ds_test, model, test_real, threshold)
    df_predict, one_line, test_images = predict_patched(df_test, df_predict_patch, patch_threshold, threshold, img_size, patches, patches_x, patches_y, patch_images)

    evaluate_simple.display_metrics(df_predict, threshold, report_path)
    metrics = evaluate_simple.get_metrics(df_predict, pred_proba_scores=False)

    plot_false_predictions(df_predict, df_predict_patch, patches_x, patches_y, threshold, report_path)
    plot_mean_probabilites(df_predict_patch, one_line, patches, patches_x, patches_y, threshold, report_path)
    plot_probabilities(df_predict_patch, one_line_patch, threshold, report_path)
    plot_patching(df_test, df_predict, df_predict_patch, test_images, patches, patches_x, patches_y, threshold, report_path)

    return metrics, df_predict, df_predict_patch
