import json
import keras.utils as utils
from keras.utils import image_dataset_from_directory
import numpy as np
import keras.callbacks as callbacks
import keras.saving as saving
import pandas as pd
import os
import glob
import create_model
import visualize_train

config_file = os.environ["CONFIG_FILE"]
data_train_dir = os.environ["DATA_TRAIN_DIR"]
data_test_dir = os.environ["DATA_TEST_DIR"]
preprocessing_report_file = os.environ["PREPROCESSING_REPORT_FILE"]
reports_dir = os.environ["REPORTS_DIR"]
model_file = os.environ["MODEL_FILE"]
training_report_file = os.environ["TRAINING_REPORT_FILE"]

with open(config_file, "r") as f:
    config = json.load(f)

with open(preprocessing_report_file, "r") as f:
    report = json.load(f)

os.makedirs(reports_dir, exist_ok=True)

params = {
    "learning_rate": 0.001,
    "augmentations": { },
    "center_scaled": False,
    "conv_blocks": [
        {
            "filters": 32,
            "dropout": 0,
            "normalization": False
        },
        {
            "filters": 64,
            "dropout": 0.1,
            "normalization": False
        },
        {
            "filters": 128,
            "dropout": 0.2,
            "normalization": False
        }
    ],
    "dense_blocks": [
        {
            "units": 128,
            "l2": 0.01,
            "dropout": 0,
            "normalization": False
        }
    ],
    "random_state": 42
}
params.update(config["modeling"])
params["augmentations"] = {
    "flip": False,
    "brightness": 0,
    "contrast": 0,
    "saturation": [0.5, 0.5],
    "hue": 0
}
params["augmentations"].update(config["modeling"]["augmentations"])

report["modeling"] = {}
rep = report["modeling"]
rep["params"] = params

img_size = report["preprocessing"]["params"]["patch_size"]
grayscale = report["grayscale"]

utils.set_random_seed(params["random_state"])

model = create_model.create_model(**params, img_size=img_size, grayscale=grayscale)
model.summary()

# print(json.dumps(report, indent=2))

params = {
    "batch_size": 32,
    "epochs": 50,
    "validation_from_train": False,
    "use_buggy_early_stopping_restore_best_weights": False,
    "early_stopping": { },
    "reduce_learning_rate_on_plateau": { },
    "random_state": 42
}
params.update(config["training"])
params["early_stopping"] = {
    "min_delta": 0.01,
    "patience": 5
}
params["early_stopping"].update(config["training"]["early_stopping"])
params["reduce_learning_rate_on_plateau"] = {
    "min_delta": 0.01,
    "patience": 3,
    "factor": 0.25,
    "cooldown": 2
}
params["reduce_learning_rate_on_plateau"].update(config["training"]["reduce_learning_rate_on_plateau"])

batch_size = params["batch_size"]
validation_from_train = params["validation_from_train"]
random_state = params["random_state"]

if validation_from_train:
    ds_train, ds_val = image_dataset_from_directory(
        directory = data_train_dir,
        image_size=(img_size, img_size),
        validation_split = 0.2,
        subset = "both",
        seed = random_state,
        batch_size = batch_size,
        shuffle = True,
        label_mode = "binary",
        color_mode = "grayscale" if grayscale else "rgb"
    )
else:
    ds_train = image_dataset_from_directory(
        directory = data_train_dir,
        image_size=(img_size, img_size),
        seed = random_state,
        batch_size = batch_size,
        shuffle = True,
        label_mode = "binary",
        color_mode = "grayscale" if grayscale else "rgb"
    )
    ds_val = image_dataset_from_directory(
        directory = data_test_dir,
        image_size=(img_size, img_size),
        seed = random_state,
        batch_size = batch_size,
        shuffle = True,
        label_mode = "binary",
        color_mode = "grayscale" if grayscale else "rgb"
    )

epoch_sum = 0

monitor = "val_loss" if validation_from_train else "loss"

early_stopping_params = params["early_stopping"]
restore_best = params["use_buggy_early_stopping_restore_best_weights"]
early_stopping = callbacks.EarlyStopping(monitor=monitor, restore_best_weights=restore_best, verbose=1, **early_stopping_params)

reduce_lr_params = params["reduce_learning_rate_on_plateau"]
reduce_lr = callbacks.ReduceLROnPlateau(monitor=monitor, verbose=1, **reduce_lr_params)

model_file_name = os.path.join(reports_dir, "model_{epoch}.keras") if restore_best else model_file
print(model_file_name)
model_checkpoint = callbacks.ModelCheckpoint(model_file_name, monitor=monitor, verbose=0, save_best_only=True, save_freq="epoch")

epochs = params["epochs"]
# epochs = 2
print("overall epochs", epoch_sum)

model_history = model.fit(ds_train, validation_data=ds_val, epochs=epochs, callbacks=[early_stopping, reduce_lr, model_checkpoint])

# epochs = len(model_history.epoch)
epoch_sum += len(model_history.epoch)

if restore_best:
    best_epoch = early_stopping.best_epoch + 1
else:
    best_epoch = int(np.argmin(model_history.history["loss"])) + 1
print("best epoch", best_epoch)
visualize_train.plot_history(model_history.history, best_epoch, len(model_history.epoch), os.path.join(reports_dir, "training_history.png"), validation_from_train)

if restore_best:
    early_stopping_best_model = saving.load_model(model_file_name.format(epoch=best_epoch))
    saving.save_model(early_stopping_best_model, model_file, overwrite=True)
    files = glob.glob(os.path.join(reports_dir, "model_*.keras"))
    for f in files:
        os.remove(f)

df_history = pd.DataFrame(model_history.history)
df_history["epoch"] = model_history.epoch
df_history.epoch += 1
df_history["validation_set"] = "train_split" if validation_from_train else "test"
df_history["monitor"] = "val_loss" if validation_from_train else "loss"
df_history.to_csv(os.path.join(reports_dir, "training_history.csv"))

report["training"] = {}
rep = report["training"]
rep["params"] = params
rep["metrics"] = {}
rep = rep["metrics"]
rep["epochs"] = len(model_history.epoch)
rep["best_epoch"] = best_epoch
rep["scores"] = {
    "accuracy": np.round(model_history.history["accuracy"][best_epoch - 1], 3),
    "loss": np.round(model_history.history["loss"][best_epoch - 1], 4),
    "val_accuracy": np.round(model_history.history["val_accuracy"][best_epoch - 1], 3),
    "val_loss": np.round(model_history.history["val_loss"][best_epoch - 1], 4)
}

# print(json.dumps(report["training"], indent=2))

with open(training_report_file, mode="w") as f:
    json.dump(report, f, indent=2)
