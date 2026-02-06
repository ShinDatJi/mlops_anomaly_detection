import keras.optimizers as optimizers
import keras.layers as layers
import keras.regularizers as regularizers
import keras.metrics as metrics
import keras.losses as losses
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers import Rescaling
from keras.layers import RandomFlip, RandomRotation, RandomBrightness, RandomContrast, RandomTranslation, RandomColorJitter
from keras.layers import BatchNormalization

def create_augmentations(x, random_state, grayscale, flip=False, brightness=0, contrast=0, saturation=(0.5, 0.5), hue=0):
    if flip == True:
        x = RandomFlip("horizontal_and_vertical", seed=random_state, name="rand_flip")(x)
    elif flip:
        x = RandomFlip(flip, seed=random_state, name="rand_flip")(x)
    if grayscale:
        if brightness:
            x = RandomBrightness(brightness, value_range=(0, 255), seed=random_state, name="rand_brightness")(x)
        if contrast:
            x = RandomContrast(contrast, value_range=(0, 255), seed=random_state, name="rand_contrast")(x)
    else:
        x = RandomColorJitter(value_range=(0, 255), brightness_factor=brightness, contrast_factor=contrast, saturation_factor=saturation, hue_factor=hue, seed=random_state, name="rand_color")(x)
    return x

def create_conv_block(x, i, random_state, filters, kernel_size=3, pool_size=2, normalization=False, dropout=False):
    x = Conv2D(filters=filters, kernel_size=kernel_size, padding="valid", activation=None, kernel_initializer="he_uniform", name="conv_block" + str(i + 1) + "_conv")(x)
    if normalization:
        x = BatchNormalization(name="conv_block" + str(i + 1) + "_norm")(x)
    x = layers.ReLU(name="conv_block" + str(i + 1) + "_relu")(x)
    if dropout:
        x = layers.SpatialDropout2D(dropout, seed=random_state, name="conv_block" + str(i + 1) + "_dropout")(x)
    x = MaxPooling2D(pool_size=pool_size, name="conv_block" + str(i + 1) + "_pool")(x)
    return x

def create_dense_block(x, i, random_state, units, l1=0, l2=0, normalization=False, dropout=False):
    x = Dense(units=units, activation=None, kernel_initializer="he_uniform", kernel_regularizer=regularizers.l1_l2(l1, l2), name="dense_block" + str(i + 1) + "_dense")(x)
    if normalization:
        x = BatchNormalization(name="dense_block" + str(i + 1) + "_norm")(x)
    x = layers.ReLU(name="dense_block" + str(i + 1) + "_relu")(x)
    if dropout:
        x = Dropout(rate=dropout, seed=random_state, name="dense_block" + str(i + 1) + "_dropout")(x) 
    return x

def create_small_model_new(x, conv_blocks, dense_blocks, random_state):
    for i, block in enumerate(conv_blocks):
        x = create_conv_block(x, i, random_state, **block)
    x = layers.GlobalAveragePooling2D(name="global_avg_pooling")(x)
    for i, block in enumerate(dense_blocks):
        x = create_dense_block(x, i, random_state, **block)
    return x

def create_model(img_size, learning_rate, augmentations, conv_blocks, dense_blocks, random_state, grayscale, center_scaled=False):
    inputs = Input(shape=(img_size, img_size, 1 if grayscale else 3), name="input")
    x = inputs
    x = create_augmentations(x, random_state, grayscale, **augmentations)
    if center_scaled:
        x = Rescaling(1./127.5, offset=-1, name="rescaling")(x)
    else:
        x = Rescaling(1./255, name="rescaling")(x)
    x = create_small_model_new(x, conv_blocks, dense_blocks, random_state)
    # x = Dense(units=1, activation="sigmoid", name="output")(x)
    x = Dense(units=1, activation=None, name="output")(x)

    model = Model(inputs, x, name="small")
    optimizer = optimizers.Adam(learning_rate)
    f1_score = metrics.F1Score(threshold=0.5, average=None)
    loss = losses.BinaryCrossentropy(from_logits=True)
    # loss = losses.BinaryFocalCrossentropy()
    # model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy", metrics.AUC(name="roc_auc", curve="ROC"), metrics.AUC(name="pr_auc", curve="PR")])
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy", f1_score])
    return model
