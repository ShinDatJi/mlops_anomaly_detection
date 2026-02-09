import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.utils import image_dataset_from_directory
from sklearn import metrics as skmetrics
import matplotlib.pyplot as plt
import seaborn as sns
import os

def display_metrics(df_predict, threshold, report_path=None):
    # print(df_predict)
    # print(pd.crosstab(df_predict.real, df_predict.pred, rownames=["real"], colnames=["predicted"], values=[1,0]))

    # print(f"Threshold: {threshold:0.2f}", end="\n\n")
    plt.figure(figsize=(8, 3))
    # plt.suptitle(f"Threshold: {threshold:0.2f}")
    plt.subplot(1, 2, 1)
    plt.title(f"Confusion matrix (threshold: {threshold:0.2f})")
    confusion_matrix = pd.crosstab(df_predict.real, df_predict.pred, rownames=["real"], colnames=["predicted"])
    # cm = cm.reindex(index=[1, 0], columns=[1, 0])
    ax = sns.heatmap(confusion_matrix, annot=True, fmt=".0f", cbar=False, vmin=0, vmax=len(df_predict), square=True, cmap="Greens_r")
    # ax = sns.heatmap(confusion_matrix, annot=True, fmt=".0f", cbar=False, vmin=0, vmax=len(df_predict), square=True, mask=[False, True, True, False], cmap="Greens")
    ax.tick_params(axis='both', which='both', length=0)

    # print(pd.crosstab(df_predict.real, df_predict.pred, rownames=["real"], colnames=["pred"]), end="\n\n")

    plt.subplot(1, 2, 2)
    title = plt.title(f"Classification report (threshold: {threshold:0.2f})")
    # title.set_pad(20)
    precision_0 = skmetrics.precision_score(df_predict.real, df_predict.pred, pos_label=0)
    recall_0 = skmetrics.recall_score(df_predict.real, df_predict.pred, pos_label=0)
    f1_score_0 = skmetrics.f1_score(df_predict.real, df_predict.pred, pos_label=0)
    precision_1 = skmetrics.precision_score(df_predict.real, df_predict.pred, pos_label=1)
    recall_1 = skmetrics.recall_score(df_predict.real, df_predict.pred, pos_label=1)
    f1_score_1 = skmetrics.f1_score(df_predict.real, df_predict.pred, pos_label=1)
    accuracy = skmetrics.accuracy_score(df_predict.real, df_predict.pred)
    df_score = pd.DataFrame([[precision_0, recall_0, f1_score_0, accuracy], [precision_1, recall_1, f1_score_1, np.nan]], columns=["precision", "recall", "f1", "accuracy"])
    ax = sns.heatmap(df_score, annot=True, fmt="0.2f", cbar=True, vmin=0.5, vmax=1, square=True, cmap="RdYlGn")
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.tick_params(axis='both', which='both', length=0)
    # plt.xticks(rotation=45, ha="left")

    plt.tight_layout()
    if report_path:
        plt.savefig(os.path.join(report_path, "confusion_matrix.png"))
    plt.close()

    # print(skmetrics.classification_report(df_predict.real, df_predict.pred))

def get_metrics(df_predict, pred_proba_scores=True):
    metrics = {}
    metrics["tp"] = int(((df_predict.real == 1) & (df_predict.pred == 1)).sum())
    metrics["tn"] = int(((df_predict.real == 0) & (df_predict.pred == 0)).sum())
    metrics["fp"] = int(((df_predict.real == 0) & (df_predict.pred == 1)).sum())
    metrics["fn"] = int(((df_predict.real == 1) & (df_predict.pred == 0)).sum())
    metrics["accuracy"] = np.round(skmetrics.accuracy_score(df_predict.real, df_predict.pred), 3)
    metrics["precision"] = np.round(skmetrics.precision_score(df_predict.real, df_predict.pred), 3)
    metrics["recall"] = np.round(skmetrics.recall_score(df_predict.real, df_predict.pred), 3)
    metrics["f1_score"] = np.round(skmetrics.f1_score(df_predict.real, df_predict.pred), 3)
    if pred_proba_scores:
        metrics["AUC"] = np.round(skmetrics.roc_auc_score(df_predict.real, df_predict.pred_proba), 3)
        metrics["AP"] = np.round(skmetrics.average_precision_score(df_predict.real, df_predict.pred_proba), 3)

    return metrics

def plot_probabilities(df_predict, threshold, report_path):
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), width_ratios=[2, 1])
    fig = plt.figure(figsize=(10, 7))
    subfigs = fig.subfigures(1, 2, width_ratios=[2, 1])

    # ax1 = subfigs[0].subplots(1, 1)[0]
    ax1 = subfigs[0].subplots(1, 1)
    one_line = df_predict[df_predict.real == 1].index[0]
    # mean = df_predict.pred_proba.mean()
    df_sort = df_predict.sort_values(by="pred_proba")
    width = len(df_predict)
    ax1.bar(range(width), np.where(df_sort.real.values == 0, df_sort.pred_proba, 0), width=1, color="tab:blue", label="class: negative")
    ax1.bar(range(width), np.where(df_sort.real.values == 1, df_sort.pred_proba, 0), width=1, color="tab:orange", label="class: positive")
    ax1.axvline(one_line, color="red", alpha=0.5)
    ax1.axhline(threshold, color="red", alpha=0.5)
    # ax1.axhline(mean, color="red", alpha=0.5)
    ax1.text(0.02 * width, threshold + 0.01, f"threshold: {threshold:0.2f}", verticalalignment="bottom")
    # ax1.text(0.02 * width, mean + 0.01, f"mean: {mean:0.2f}", verticalalignment="bottom")
    ax1.text(one_line - 0.01 * width, 0.98, "neg/pos", horizontalalignment="right", verticalalignment="top")
    ax1.set_xlim(0, width - 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Image")
    ax1.set_ylabel("Probabilty")
    ax1.set_title("Sorted probability bar plot")
    ax1.legend(loc="upper left")

    axs2 = subfigs[1].subplots(2, 1)
    ax21 = axs2[0]
    ax22 = axs2[1]
    # fig, (ax21, ax22) = ax2.subplots(2, 1)
    skmetrics.RocCurveDisplay.from_predictions(df_predict.real, df_predict.pred_proba, ax=ax21)
    ax21.set_title("ROC curve")
    ax21.grid(alpha=0.3)

    skmetrics.PrecisionRecallDisplay.from_predictions(df_predict.real, df_predict.pred_proba, ax=ax22)
    ax22.set_title("PR curve")
    ax22.grid(alpha=0.3)

    plt.tight_layout(pad=2.0)
    if report_path:
        plt.savefig(os.path.join(report_path, "prediction_probabilities.png"), bbox_inches="tight")
    plt.close()

def plot_grad_cam(model, df_predict, threshold, images, block_count, report_path, grayscale):

    def grad_cam(img, layer):
        img = np.expand_dims(img, axis=0)
        img = tf.convert_to_tensor(img)
        grad_model = Model(model.inputs, [model.get_layer(layer).output, model.output])
        with tf.GradientTape() as tape:
            layer_output, preds = grad_model(img, training=False)
            # channel = ([1 if preds[0] >= 0.5 else 0])
            pred_proba = preds[0]
        grads = tape.gradient(pred_proba, layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        # heatmap = layer_output[0] @ pooled_grads[..., tf.newaxis]
        # heatmap = tf.squeeze(heatmap)
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, layer_output[0]), axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        heatmap_max = tf.math.reduce_max(heatmap)
        if heatmap_max > 0:
            heatmap /= heatmap_max
        heatmap = heatmap.numpy()
        heatmap = heatmap * 255
        # print(heatmap.max())
        # print(heatmap.min())
        heatmap = heatmap.astype("uint8")
        return heatmap

    def plot_grad_cam(col, img, i, title):
        t = title + f": {df_predict.pred_proba[i]:.2f}"
        # if not patching:
        #     t += " " + df_test.index[i]
        rows = block_count + 1
        plt.subplot(rows, 8, col)
        plt.imshow(np.array(img).astype("uint8"), cmap="Grays_r" if grayscale else None)
        plt.axis("off")
        plt.title(t)
        for b in range(block_count):
            plt.subplot(rows, 8, 8 * (b + 1) + col)
            plt.imshow(grad_cam(img, "conv_block" + str(b + 1) + "_pool"))
            plt.axis("off")
            # plt.subplot(rows, 8, col + 16)
            # plt.imshow(grad_cam(img, "block2pool"))
            # plt.axis("off")
            # plt.subplot(rows, 8, col + 24)
            # plt.imshow(grad_cam(img, "block3pool"))
            # plt.axis("off")

    plt.figure(figsize=(12, block_count * 2))
    plt.suptitle(f"Grad-CAM (threshold: {threshold:0.2f})")
    tp_count = 0
    fp_count = 0
    tn_count = 0
    fn_count = 0
    # for i, img in enumerate(test_images):
    for i in np.random.permutation(len(images)):
        img = images[i]
        if (tp_count < 2) & (df_predict.real[i] == 1) & (df_predict.pred[i] == 1):
            tp_count += 1
            plot_grad_cam(tp_count, img, i, "TP")
        if (fn_count < 2) & (df_predict.real[i] == 1) & (df_predict.pred[i] == 0):
            fn_count += 1
            plot_grad_cam(2 + fn_count, img, i, "FN")
        if (tn_count < 2) & (df_predict.real[i] == 0) & (df_predict.pred[i] == 0):
            tn_count += 1
            plot_grad_cam(4 + tn_count, img, i, "TN")
        if (fp_count < 2) & (df_predict.real[i] == 0) & (df_predict.pred[i] == 1):
            fp_count += 1
            plot_grad_cam(6 + fp_count, img, i, "FP")
    plt.tight_layout()
    if report_path:
        plt.savefig(os.path.join(report_path, "grad_cam.png"))
    plt.close()

def load_data(data_path, img_size, batch_size, grayscale):

    ds_test = image_dataset_from_directory(
        directory = data_path,
        image_size=(img_size, img_size),
        batch_size = batch_size,
        shuffle = False,
        label_mode = "binary",
        verbose=False,
        color_mode = "grayscale" if grayscale else "rgb"
    )

    # test_images = []
    # test_real = []
    # for img, t in ds_test.unbatch():
    #     test_images.append(img)
    #     test_real.append(t)
    # test_images = np.array(test_images)
    # test_real = np.array(test_real)

    # test_images = np.array([img for img, t in ds_test.unbatch()])
    # test_real = np.array([int(t) for img, t in ds_test.unbatch()])

    unbatched = list(ds_test.unbatch())
    test_images = np.array([img for img, t in unbatched])
    test_real   = np.array([int(t) for img, t in unbatched])

    return ds_test, test_images, test_real

def predict(ds_test, model, test_real, threshold):
    df_predict = pd.DataFrame(test_real, columns=["real"])

    test_pred_proba_logit = model.predict(ds_test, verbose=False)[:,0]
    test_pred_proba = tf.sigmoid(test_pred_proba_logit).numpy()
    # test_pred_proba = model.predict(ds_test, verbose=False)[:,0]

    # test_pred_proba = model(test_images, training=True)
    # test_pred_proba = model.predict(test_images, verbose=False)[:,0]
    # print(model.evaluate(test_images, test_real, verbose=False))
    df_predict["pred_proba"] = test_pred_proba

    # one_line = np.where(test_pred_score >= threshold)[0][0]
    # one_line = np.where(np.array(test_real) == 1)[0][0]
    one_line = df_predict[df_predict.real == 1].index[0]
    
    if threshold == "balanced":
        test_pred_proba_sorted = df_predict.pred_proba.sort_values().values
        threshold = (test_pred_proba_sorted[one_line] + test_pred_proba_sorted[one_line - 1]) / 2
        # threshold = (df_predict[df_predict.real == 0].pred_proba.max() + df_predict[df_predict.real == 1].pred_proba.min()) / 2
        # threshold = (test_pred_proba_sorted[one_line])
        # threshold = df_sort[df_sort[1] == 0].iloc[-1, 0]
    else:
        threshold = threshold
    test_pred = (test_pred_proba >= threshold).astype(int)
    df_predict["pred"] = test_pred

    return df_predict, one_line, threshold

def test_model(data_path, model, img_size, batch_size, block_count, threshold="balanced", report_path=None, grayscale=False):

    ds_test, test_images, test_real = load_data(data_path, img_size, batch_size, grayscale)
    df_predict, one_line, threshold = predict(ds_test, model, test_real, threshold)

    display_metrics(df_predict, threshold, report_path)
    plot_probabilities(df_predict, threshold, report_path)
    plot_grad_cam(model, df_predict, threshold, test_images, block_count, report_path, grayscale)

    metrics = get_metrics(df_predict)
    
    return metrics, threshold
