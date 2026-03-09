import cv2
import numpy as np
import tensorflow as tf
from mlflow import MlflowClient
import mlflow
import os

os.environ["MLFLOW_TRACKING_USERNAME"] = "mlflow"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "mlflowpassword"

models_params = {}
models = {}

def load_model(category: str, version: str) -> None:
    mlflow.set_tracking_uri("http://localhost:5000")
    client = MlflowClient()

    model_version = client.get_model_version_by_alias(f"{category}_{version}", "champion")
    run = client.get_run(model_version.run_id)
    params = run.data.params
    model = mlflow.keras.load_model(f"models:/{category}_{version}@champion")
    if category not in models:
        models[category] = model
        models_params[category] = params

def load_image(image_bin: bytes, grayscale: bool):
    image_np = np.frombuffer(image_bin, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR_RGB)
    if image is None:
        raise ValueError("Unable to decode image")
    if grayscale:
        image = image[:, :, :1]
    return image


def create_patches(img, patch_size, patches, overlap, height_cropping, width_cropping):
    img_size = img.shape[0]

    step = int(img_size / (patches + overlap / (1 - overlap)))
    patch_size_overlap = int(step / (1 - overlap))
    new_img_size = (patches - 1) * step + patch_size_overlap
    start = (img_size - new_img_size) // 2

    patch_images = []
    for p_r in range(height_cropping, patches - height_cropping):
        r = start + p_r * step
        for p_c in range(width_cropping, patches - width_cropping):
            c = start + p_c * step
            patch = img[r : r + patch_size_overlap, c : c + patch_size_overlap, :]
            patch = cv2.resize(patch, dsize=(patch_size, patch_size))
            patch_images.append(patch)

    patch_images = np.array(patch_images)

    return tf.convert_to_tensor(patch_images)


def predict_patched(model, images, threshold):
    logits = model.predict(images, verbose=False)[:, 0]
    pred_probas = tf.sigmoid(logits).numpy()
    pred = int(pred_probas.max() >= threshold)
    pred_probas = pred_probas.tolist()

    return pred, pred_probas


def predict(category: str, version: str, image_bin: bytes) -> int:
    load_model(category, version)

    grayscale = models_params[category]["grayscale"] == "True"
    patch_size = int(models_params[category]["preprocessing_patch_size"])
    patches = int(models_params[category]["preprocessing_patches"])
    overlap = float(models_params[category]["preprocessing_overlap"])
    height_cropping = int(models_params[category]["preprocessing_height_cropping"])
    width_cropping = int(models_params[category]["preprocessing_width_cropping"])
    threshold = float(models_params[category]["evaluation_threshold"])

    image = load_image(image_bin, grayscale)
    images = create_patches(image, patch_size, patches, overlap, height_cropping, width_cropping)
    pred, pred_probas = predict_patched(models[category], images, threshold)

    params = {
        "patches": patches,
        "overlap": overlap,
        "height_cropping": height_cropping,
        "width_cropping": width_cropping,
        "threshold": threshold
    }

    return pred, pred_probas, params