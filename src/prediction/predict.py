import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
import keras.saving as saving

test_file = "./data/mad_test.csv"
model_dir = "./models/"

def create_patches(img, config):
    config = config["preprocessing"]
    img_size = img.shape[0]
    patches = config["patches"]
    overlap = config["overlap"]
    embedding = config["embedding"]
    height_cropping = config["height_cropping"]
    if "width_cropping" in config:
        width_cropping = config["width_cropping"]
    else:
        width_cropping = 0
    patch_size = config["patch_size"]

    step = int(img_size / (patches + overlap / (1 - overlap)))
    patch_size_overlap = int(step / (1 - overlap))
    new_img_size = (patches - 1) * step + patch_size_overlap
    start = (img_size - new_img_size) // 2

    patch_images = []
    if embedding:
        img_small = cv2.resize(img, dsize=(embedding, embedding))
    for p_r in range(height_cropping, patches - height_cropping):
        r = start + p_r * step
        for p_c in range(width_cropping, patches - width_cropping):
            c = start + p_c * step
            patch = img[r:r + patch_size_overlap, c:c + patch_size_overlap, :]
            patch = cv2.resize(patch, dsize=(patch_size, patch_size))
            if embedding:
                patch[0:embedding, 0:embedding] = img_small
            patch_images.append(patch)

    return np.array(patch_images)

def predict_patched(model, image, config):
    images = create_patches(image, config)
    images = tf.convert_to_tensor(images)
    logits = model.predict(images, verbose=False)[:,0]
    pred_proba = tf.sigmoid(logits).numpy()

    threshold = config["testing"]["patch_threshold_mode"]

    return int(pred_proba.max() >= threshold), pred_proba

def plot_patching(image, config, pred, real, pred_proba, file_name, ground_truth=None):
    patch_images = create_patches(image, config)
    threshold = config["testing"]["patch_threshold_mode"]

    config = config["preprocessing"]
    patches = config["patches"]
    patch_size = config["patch_size"]
    height_cropping = config["height_cropping"]
    if "width_cropping" in config:
        width_cropping = config["width_cropping"]
    else:
        width_cropping = 0

    patches_x = patches - 2 * width_cropping
    patches_y = patches - 2 * height_cropping

    patch_images = patch_images.reshape((patches_y, patches_x, patch_size, patch_size, -1))
    patch_image = np.zeros((patches_y * patch_size, patches_x * patch_size, 3))
    for r in range(patches_y):
        for c in range(patches_x):
            patch_image[r*patch_size:r*patch_size+patch_size, c*patch_size:c*patch_size+patch_size] = patch_images[r, c, :, :, :]

    fig = plt.figure(figsize=(12, 3))
    # plt.suptitle("Prediction: " + ("Defective" if pred else "Good"), x=0.1)
    fig.patch.set_linewidth(5)
    fig.patch.set_edgecolor("Red" if pred != real else "Green")

    ax = plt.subplot(1, 4, 1)
    plt.imshow(image, cmap="Grays_r" if image.shape[2] == 1 else None)
    plt.axis("off")
    plt.text(0, image.shape[0] * 1.075, file_name, fontsize="xx-small")
    plt.title("Prediction: " + ("Defective" if pred else "Good") + "\n" + "Real: " + ("Defective" if real else "Good"), color="Red" if pred != real else "Green")

    plt.subplot(1, 4, 2)
    plt.imshow(ground_truth, cmap="Grays_r")
    plt.axis("off")
    plt.title("Ground truth")

    plt.subplot(1, 4, 3)
    plt.imshow(np.array(patch_image).astype("uint8"))
    plt.hlines(range(patch_image.shape[0] // patches_y, patch_image.shape[0], patch_image.shape[0] // patches_y), xmin=0, xmax=patch_image.shape[1] - 1, color="blue", linestyle="-", linewidth=1, alpha=0.5)
    plt.vlines(range(patch_image.shape[1] // patches_x, patch_image.shape[1], patch_image.shape[1] // patches_x), ymin=0, ymax=patch_image.shape[0] - 1, color="blue", linestyle="-", linewidth=1, alpha=0.5)
    for p in range(patches_x * patches_y):
        if pred_proba[p] >= threshold:
            x_offset = (patch_image.shape[1] // patches_x)
            y_offset = (patch_image.shape[0] // patches_y)
            x = (p % patches_x) * x_offset
            y = (p // patches_x) * y_offset
            xs = [x, x + x_offset - 1, x + x_offset - 1, x, x]
            ys = [y, y, y + y_offset - 1, y + y_offset - 1, y]
            plt.fill(xs, ys, color="red", alpha=0.25)
            plt.plot(xs, ys, color="red", linewidth=1, linestyle="-", alpha=0.5)
    plt.axis("off")
    plt.title("Patch activation map")

    plt.subplot(1, 4, 4)
    sns.heatmap(pred_proba.reshape((patches_y, patches_x)), annot=True, square=True, vmin=0, vmax=1, cbar=False, fmt="0.1f")
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(f"threshold: {threshold:0.2f}")
    plt.title("Patch probability map")

    plt.tight_layout()

    return fig

def predict_random(category, defective):
    df_test = pd.read_csv(test_file, index_col=0)
    # df_test["file"] = df_test.file.str[3:]
    # df_test["file_ground_truth"] = df_test.file_ground_truth.str[3:]
    df_test = df_test[df_test.category == category]
    if defective:
        df_test = df_test[df_test.anomaly != "good"]
    else:
        df_test = df_test[df_test.anomaly == "good"]
    with open(model_dir + category + "/config.json", "r") as f:
        config = json.load(f)
    model = saving.load_model(model_dir + category + "/model.keras")

    num = np.random.randint(len(df_test))

    real = int(df_test.anomaly.iloc[num] != "good")
    image = cv2.imread(df_test.file.iloc[num], cv2.IMREAD_COLOR_RGB)
    if df_test.grayscale.iloc[0]:
        image = image[:, :, :1]
    pred, pred_proba = predict_patched(model, image, config)

    if df_test.anomaly.iloc[num] != "good":
        ground_truth = cv2.imread(df_test.file_ground_truth.iloc[num])
    else:
        ground_truth = np.zeros((image.shape[0], image.shape[1]))

    return plot_patching(image, config, pred, real, pred_proba, df_test.file.iloc[num], ground_truth)
    
fig = predict_random("bottle", True)
plt.show(block=True)

fig = predict_random("bottle", False)
plt.show(block=True)