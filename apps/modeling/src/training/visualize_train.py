import matplotlib.pyplot as plt

def plot_history(history, best_epoch, epochs, out_file, validation_from_train):
    plt.figure(figsize=(8,4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history["loss"], label="train set")
    label = "validation set" if validation_from_train else "test set"
    plt.plot(range(1, epochs + 1), history["val_loss"], label=label)
    color = "tab:orange" if validation_from_train else "tab:blue"
    plt.axvline(best_epoch, color=color, alpha=0.5)
    key = "val_loss" if validation_from_train else "loss"
    txt = "(best early stopping)" if validation_from_train else "(best)"
    plt.text(best_epoch - 0.01 * epochs, 0.98, f"{history[key][best_epoch - 1]:0.4f}" + txt, horizontalalignment="right", verticalalignment="top")
    plt.xlim([0, epochs + 1])
    plt.ylim(0, 1)
    plt.xlabel("epoch")
    plt.legend()
    plt.title("Learning Curve - Loss")

    # plt.subplot(1, 2, 2)
    # plt.plot(range(1, epochs + 1), history["accuracy"], label="train set")
    # plt.plot(range(1, epochs + 1), history["val_accuracy"], label=label)
    # plt.axvline(best_epoch, color=color, alpha=0.5)
    # key = "val_accuracy" if validation_from_train else "accuracy"
    # plt.text(best_epoch - 0.01 * epochs, 0.02, f"{history[key][best_epoch - 1]:0.4f}", horizontalalignment="right", verticalalignment="bottom")
    # plt.xlim([0, epochs + 1])
    # plt.ylim(0, 1)
    # plt.xlabel("epoch")
    # plt.legend()
    # plt.title("Learning curve - Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), history["f1_score"], label="train set")
    plt.plot(range(1, epochs + 1), history["val_f1_score"], label=label)
    plt.axvline(best_epoch, color=color, alpha=0.5)
    key = "val_f1_score" if validation_from_train else "f1_score"
    plt.text(best_epoch - 0.01 * epochs, 0.02, f"{history[key][best_epoch - 1]:0.4f}", horizontalalignment="right", verticalalignment="bottom")
    plt.xlim([0, epochs + 1])
    plt.ylim(0, 1)
    plt.xlabel("epoch")
    plt.legend()
    plt.title("Learning curve - F1-score")

    # plt.subplot(2, 2, 3)
    # plt.plot(history["roc_auc"], label="roc auc")
    # plt.plot(history["val_roc_auc"], label="val_roc_auc")
    # plt.xlim([0, epochs])
    # plt.legend()
    # plt.title("ROC AUC")

    # plt.subplot(2, 2, 4)
    # plt.plot(history["pr_auc"], label="pr auc")
    # plt.plot(history["val_pr_auc"], label="val_pr_auc")
    # plt.xlim([0, epochs])
    # plt.legend()
    # plt.title("PR AUC")

    plt.tight_layout()
    if out_file:
        plt.savefig(out_file)
    plt.close()