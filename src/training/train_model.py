import json
import keras.utils as utils
from keras.utils import image_dataset_from_directory
import numpy as np
import keras.callbacks as callbacks
import keras.saving as saving
from pathlib import Path
import pandas as pd
import create_model
import visualize_train

category = "bottle"

config_file = f"./models/{category}/config.json"
preprocessing_report_file = f"./reports/preprocessing/{category}/report.json"
data_processed_train_dir = f"./data/processed/{category}/train/"
data_processed_test_dir = f"./data/processed/{category}/test/"
reports_models_dir = f"./reports/models/{category}/"
reports_training_dir = f"./reports/training/{category}/"
training_report_file = f"./reports/training/{category}/report.json"

with open(config_file, "r") as f:
    config = json.load(f)

with open(preprocessing_report_file, "r") as f:
    report = json.load(f)

Path(reports_models_dir).mkdir(parents=True, exist_ok=True)
Path(reports_training_dir).mkdir(parents=True, exist_ok=True)

random_state = config["random_state"]
utils.set_random_seed(random_state)

params = config["modeling"]

report["modeling"] = {}
rep = report["modeling"]
rep["params"] = params

img_size = report["preprocessing"]["params"]["patch_size"]
grayscale = report["metrics"]["grayscale"]

model = create_model.create_model(**params, img_size=img_size, seed=random_state, grayscale=grayscale)
model.summary()

# print(json.dumps(report, indent=2))


params = config["training"]
batch_size = params["batch_size"]
if "validation_from_train" in params:
    validation_from_train = params["validation_from_train"]
else:
    validation_from_train = False

if validation_from_train:
    ds_train, ds_val = image_dataset_from_directory(
        directory = data_processed_train_dir,
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
        directory = data_processed_train_dir,
        image_size=(img_size, img_size),
        seed = random_state,
        batch_size = batch_size,
        shuffle = True,
        label_mode = "binary",
        color_mode = "grayscale" if grayscale else "rgb"
    )
    ds_val = image_dataset_from_directory(
        directory = data_processed_test_dir,
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
if "use_buggy_early_stopping_restore_best_weights" in params:
    use_buggy_early_stopping_restore_best_weights = params["use_buggy_early_stopping_restore_best_weights"]
else:
    use_buggy_early_stopping_restore_best_weights = False
restore_best = use_buggy_early_stopping_restore_best_weights
early_stopping = callbacks.EarlyStopping(monitor=monitor, restore_best_weights=restore_best, verbose=1, **early_stopping_params)

reduce_lr_params = params["reduce_learning_rate_on_plateau"]
reduce_lr = callbacks.ReduceLROnPlateau(monitor=monitor, verbose=1, **reduce_lr_params)

file_name = "model_{epoch}.keras" if use_buggy_early_stopping_restore_best_weights else "model.keras"
model_checkpoint = callbacks.ModelCheckpoint(reports_models_dir + file_name, monitor=monitor, verbose=0, save_best_only=True, save_freq="epoch")

epochs = params["epochs"]
# epochs = 1
print("overall epochs", epoch_sum)

model_history = model.fit(ds_train, validation_data=ds_val, epochs=epochs, callbacks=[early_stopping, reduce_lr, model_checkpoint])

# epochs = len(model_history.epoch)
epoch_sum += len(model_history.epoch)

if use_buggy_early_stopping_restore_best_weights:
    best_epoch = early_stopping.best_epoch + 1
else:
    best_epoch = int(np.argmin(model_history.history["loss"])) + 1
print(best_epoch)
visualize_train.plot_history(model_history.history, best_epoch, len(model_history.epoch), reports_training_dir + "training_", validation_from_train)

if use_buggy_early_stopping_restore_best_weights:
    early_stopping_best_model = saving.load_model(reports_models_dir + "model_{}.keras".format(best_epoch))
    saving.save_model(early_stopping_best_model, reports_models_dir + "model.keras", overwrite=True)
    files = Path(reports_models_dir).glob("model_*.keras")
    for f in files:
        f.unlink()

df_history = pd.DataFrame(model_history.history)
df_history["epoch"] = model_history.epoch
df_history.epoch += 1
df_history["validation_set"] = "train_split" if validation_from_train else "test"
df_history["monitor"] = "val_loss" if validation_from_train else "loss"
df_history.to_csv(reports_training_dir + "training_history.csv")

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

with open(reports_models_dir + "config.json", mode="w") as f:
    json.dump(config, f, indent=2)
