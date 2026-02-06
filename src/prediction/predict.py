import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 0=all, 1=info, 2=warn, 3=error
import cv2
import tensorflow as tf
import numpy as np
import json
import keras.saving as saving

models_dir = "./models/"

reports = {}
models = {}

def load_files(category):
    path = os.path.join(models_dir, category)
    if not category in reports:
        with open(os.path.join(path, "report.json"), "r") as f:
            reports[category] = json.load(f)
    if not category in models:
        models[category] = saving.load_model(os.path.join(path, "model.keras"))

def load_image(image_bin, grayscale):
    image_np = np.frombuffer(image_bin, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR_RGB)
    if grayscale:
        image = image[:,:,:1]
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
            patch = img[r:r + patch_size_overlap, c:c + patch_size_overlap, :]
            patch = cv2.resize(patch, dsize=(patch_size, patch_size))
            patch_images.append(patch)

    patch_images = np.array(patch_images)

    return tf.convert_to_tensor(patch_images)

def predict_patched(model, images, threshold, patches, height_cropping, width_cropping):
    logits = model.predict(images, verbose=False)[:,0]
    pred_probas = tf.sigmoid(logits).numpy()

    patches_x = patches - (2 * width_cropping)
    patches_y = patches - (2 * height_cropping)
    pred = int(pred_probas.max() >= threshold)
    pred_probas = pred_probas.reshape((patches_y, patches_x)).tolist()

    return pred, pred_probas

def predict(category, image_bin):
    load_files(category)
    image = load_image(image_bin, reports[category]["grayscale"])
    report = reports[category]
    rep = report["preprocessing"]["params"]
    patch_size = rep["patch_size"]
    patches = rep["patches"]
    overlap = rep["overlap"]
    height_cropping = rep["height_cropping"]
    width_cropping = rep["width_cropping"]
    threshold = report["evaluation"]["params"]["threshold"]   

    images = create_patches(image, patch_size, patches, overlap, height_cropping, width_cropping)
    pred, pred_probas = predict_patched(models[category], images, threshold, patches, height_cropping, width_cropping)

    return pred