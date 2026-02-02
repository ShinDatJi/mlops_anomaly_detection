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

data_dir = "./data/processed/"
model_dir = "./models/"
report_dir = "./reports/models/"
category = "bottle"

load_path = model_dir + category + "/"
with open(load_path + "config.json", "r") as f:
    config = json.load(f)

report_path = report_dir + category + "/temp/"
with open(report_path + "report.json", "r") as f:
    report = json.load(f)

model_path = model_dir + category + "/"

random_state = config["random_state"]
utils.set_random_seed(random_state)

params = config["modeling"]

report["modeling"] = {}
rep = report["modeling"]
rep["params"] = params

if report["patching"]:
    img_size = report["preprocessing"]["params"]["patch_size"]
else:
    img_size = report["preprocessing"]["params"]["image_size"]
grayscale = report["grayscale"]

model = create_model.create_model(**params, img_size=img_size, seed=random_state, grayscale=grayscale)
model.summary()

# print(json.dumps(report, indent=2))

batch_size = config["training"]["batch_size"]
if "validation_from_train" in config["training"]:
    validation_from_train = config["training"]["validation_from_train"]
else:
    validation_from_train = False

if validation_from_train:
    ds_train, ds_val = image_dataset_from_directory(
        directory = data_dir + category + "/train",
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
        directory = data_dir + category + "/train",
        image_size=(img_size, img_size),
        seed = random_state,
        batch_size = batch_size,
        shuffle = True,
        label_mode = "binary",
        color_mode = "grayscale" if grayscale else "rgb"
    )
    ds_val = image_dataset_from_directory(
        directory = data_dir + category + "/test",
        image_size=(img_size, img_size),
        seed = random_state,
        batch_size = batch_size,
        shuffle = True,
        label_mode = "binary",
        color_mode = "grayscale" if grayscale else "rgb"
    )

epoch_sum = 0

monitor = "val_loss" if validation_from_train else "loss"

early_stopping_params = config["training"]["early_stopping"]
if "use_buggy_early_stopping_restore_best_weights" in config["training"]:
    use_buggy_early_stopping_restore_best_weights = config["training"]["use_buggy_early_stopping_restore_best_weights"]
else:
    use_buggy_early_stopping_restore_best_weights = False
restore_best = True if use_buggy_early_stopping_restore_best_weights else False
early_stopping = callbacks.EarlyStopping(monitor=monitor, restore_best_weights=restore_best, verbose=1, **early_stopping_params)

reduce_lr_params = config["training"]["reduce_learning_rate_on_plateau"]
reduce_lr = callbacks.ReduceLROnPlateau(monitor=monitor, verbose=1, **reduce_lr_params)

file_name = "model_{epoch}.keras" if use_buggy_early_stopping_restore_best_weights else "model.keras"
model_checkpoint = callbacks.ModelCheckpoint(model_path + file_name, monitor=monitor, verbose=0, save_best_only=True, save_freq="epoch")

epochs = config["training"]["epochs"]
class_weight = config["training"]["class_weight"]
print("overall epochs", epoch_sum)

# model_history = model.fit(ds_train, validation_data=ds_val, epochs=epochs, class_weight={0: 1 - class_weight, 1: class_weight})
model_history = model.fit(ds_train, validation_data=ds_val, epochs=epochs, callbacks=[early_stopping, reduce_lr, model_checkpoint])
# model_history = model.fit(ds_train, epochs=epochs, callbacks=[early_stopping])

# epochs = len(model_history.epoch)
epoch_sum += len(model_history.epoch)

if use_buggy_early_stopping_restore_best_weights:
    best_epoch = early_stopping.best_epoch + 1
else:
    best_epoch = int(np.argmin(model_history.history["loss"])) + 1
print(best_epoch)
visualize_train.plot_history(model_history.history, best_epoch, len(model_history.epoch), report_path + "0_training_", validation_from_train)

if use_buggy_early_stopping_restore_best_weights:
    early_stopping_best_model = saving.load_model(model_path + "model_{}.keras".format(best_epoch))
    saving.save_model(early_stopping_best_model, model_path + "model.keras", overwrite=True)
    files = Path(model_path).glob("model_*.keras")
    for f in files:
        f.unlink()

df_history = pd.DataFrame(model_history.history)
df_history["epoch"] = model_history.epoch
df_history.epoch += 1
df_history["validation_set"] = "train_split" if validation_from_train else "test"
df_history["monitor"] = "val_loss" if validation_from_train else "loss"
df_history.to_csv(report_path + "0_training_history.csv")

params = {}
params["batch_size"] = batch_size
params["validation_from_train"] = validation_from_train
params["epochs"] = epochs
params["class_weight"] = class_weight
params["use_buggy_early_stopping_restore_best_weights"] = use_buggy_early_stopping_restore_best_weights
params["early_stopping"] = early_stopping_params
params["reduce_learning_rate_on_plateau"] = reduce_lr_params

report["training"] = {}
rep = report["training"]
rep["params"] = params
rep["epochs"] = len(model_history.epoch)
rep["best_epoch"] = best_epoch
rep["scores"] = {
    "accuracy": np.round(model_history.history["accuracy"][best_epoch - 1], 3),
    "loss": np.round(model_history.history["loss"][best_epoch - 1], 4),
    "val_accuracy": np.round(model_history.history["val_accuracy"][best_epoch - 1], 3),
    "val_loss": np.round(model_history.history["val_loss"][best_epoch - 1], 4)
}

# print(json.dumps(report["training"], indent=2))

with open(report_path + "report.json", mode="w") as f:
    json.dump(report, f, indent=2)
