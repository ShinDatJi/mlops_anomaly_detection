import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests

api_port = os.environ["MY_API_PORT"]
test_key = os.environ["MY_API_TEST_KEY"]

def predict(category, version, image_file):
    url = f"http://localhost:{api_port}/predict/{category}/{version}"
    response = requests.post(url, files={'image': open(image_file, 'rb')}, headers={"x-api-key": test_key})
    if(response.status_code != 200):
        raise ValueError(f"Prediction API error: {response.status_code} - {response.text}")
    return response.json()

def plot_patching(image, params, pred, pred_probas):
    def get_patch_params(img_size, patches, overlap):
        step = int(img_size / (patches + overlap / (1 - overlap)))
        patch_size_overlap = int(step / (1 - overlap))
        new_img_size = (patches - 1) * step + patch_size_overlap
        start = (img_size - new_img_size) // 2
        return start, step, patch_size_overlap, new_img_size

    def add_patches_(xs, ys, color, alpha, border=True):
        plt.fill(xs, ys, color=color, alpha=alpha, linestyle="-" if border else "")

    def add_overlap_patches(img_size, patches, width_cropping, height_cropping, overlap, color, mask, alphas):
        start, step, patch_size_overlap, new_img_size = get_patch_params(img_size, patches, overlap)
        patches_x = patches - 2 * width_cropping
        for r in range(height_cropping, patches - height_cropping):
            for c in range(width_cropping, patches - width_cropping):
                if mask[(r - height_cropping) * patches_x + (c - width_cropping)]:
                    x1 = start + step * c
                    x2 = start + step * c + patch_size_overlap - 1
                    y1 = start + step * r
                    y2 = start + step * r + patch_size_overlap - 1 
                    xs = [x1, x2, x2, x1, x1]
                    ys = [y1, y1, y2, y2, y1]
                    add_patches_(xs, ys, color, alphas[(r - height_cropping) * patches_x + (c - width_cropping)], False)

    patches = params["patches"]
    overlap = params["overlap"]
    height_cropping = params["height_cropping"]
    width_cropping = params["width_cropping"]
    threshold = params["threshold"]
    img_size = image.shape[0]

    patches_x = patches - 2 * width_cropping
    patches_y = patches - 2 * height_cropping

    fig = plt.figure(figsize=(12, 3), layout="constrained")
    fig.patch.set_linewidth(5)
    fig.patch.set_edgecolor("Red" if pred else "Green")

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap="Grays_r" if image.shape[2] == 1 else None)
    plt.axis("off")
    plt.title("Image")

    plt.subplot(1, 3, 2)
    plt.imshow(image)
    add_overlap_patches(img_size, patches, width_cropping, height_cropping, overlap, "red", pred_probas >= threshold, pred_probas * 0.5)
    plt.axis("off")
    plt.title("Patch activation map")

    plt.subplot(1, 4, 4)
    sns.heatmap(pred_probas.reshape((patches_y, patches_x)), annot=True, square=True, vmin=0, vmax=1, cbar=False, fmt="0.1f")
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(f"threshold: {threshold:0.2f}")
    plt.title("Patch probability map")

    # plt.tight_layout()

    return fig

def predict_random(category, version, defective):
    data_path = f"./data/raw/{category}/test"
    image_files = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".png"):
                image_files.append(os.path.join(root, file))
    if(defective):
        image_files = [f for f in image_files if "good" not in f]
    else: 
        image_files = [f for f in image_files if "good" in f]

    num = np.random.randint(len(image_files))

    image = cv2.imread(image_files[num], cv2.IMREAD_COLOR_RGB)
    res = predict(category, version, image_files[num])

    return plot_patching(image, res["params"], res["defective"], np.array(res["pred_probas"]))
    