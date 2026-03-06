import os
import json
from airflow.models import Param

default_config_file = os.environ["DEFAULT_CONFIG_FILE"]
reports_path = os.environ["REPORTS_MODELING_PATH"]
reports_config = os.environ["REPORTS_CONFIG"]

def load_default_config():
    with open(default_config_file, "r") as f:
        config = json.load(f)
    return config

def load_config(category):
    config_file = os.path.join(reports_path, category, reports_config)
    with open(config_file, "r") as f:
        config = json.load(f)
    return config

def save_config(category, config):
    config_file = os.path.join(reports_path, category, reports_config)
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

def get_params_from_config(config):
    config = config.copy()
    params = {}
    i = 0
    for k, v in config.items():
        params[str(i).zfill(2) + "_" + k] = v
        i += 1
    return params

def get_config_from_params(params):
    params = params.copy()
    config = {}
    for k, v in params.items():
        config[k[3:]] = v
    return config

# def get_preparation_params(config):
#     return {
#         "train_test_split": Param(config['preparation']['train_test_split'], type="number", minimum=0, maximum=1, section="Preparation"),
#         "preparation_random_state": Param(config['preparation']['random_state'], type="integer", section="Preparation")
#     }

# def get_preprocessing_params(config):
#     return {
#         "patch_size": Param(config['preprocessing']['patch_size'], type="integer", minimum=1, section="Preprocessing"),
#         "patches": Param(config['preprocessing']['patches'], type="integer", minimum=1, section="Preprocessing"),
#         "overlap": Param(config['preprocessing']['overlap'], type="number", minimum=0, maximum=1, section="Preprocessing"),
#         "good_fraction": Param(config['preprocessing']['good_fraction'], type="number", minimum=0, maximum=1, section="Preprocessing"),
#         "oversampling": Param(config['preprocessing']['oversampling'], type="boolean", section="Preprocessing"),
#         "preprocessing_threshold": Param(
#             config['preprocessing']['threshold'],
#             type="object",
#             schema={
#                 "oneOf": [
#                     {"type": "string", "enum": ["auto", "full-auto"]},
#                     {"type": "number"}
#                 ]
#             },
#             section="Preprocessing",
#         ),
#         "threshold_factor": Param(config['preprocessing']['threshold_factor'], type="number", minimum=0, section="Preprocessing"),
#         "height_cropping": Param(config['preprocessing']['height_cropping'], type="integer", minimum=0, section="Preprocessing"),
#         "width_cropping": Param(config['preprocessing']['width_cropping'], type="integer", minimum=0, section="Preprocessing"),
#         "random_trans": Param(config['preprocessing']['random_trans'], type="number", minimum=0, maximum=1, section="Preprocessing"),
#         "random_rot": Param(config['preprocessing']['random_rot'], type="number", minimum=0, maximum=1, section="Preprocessing"),
#         "random_trans_sub": Param(config['preprocessing']['random_trans_sub'], type="number", minimum=0, maximum=1, section="Preprocessing"),
#         "random_rot_sub": Param(config['preprocessing']['random_rot_sub'], type="number", minimum=0, maximum=1, section="Preprocessing"),
#         "fill_mode": Param(
#             config['preprocessing']['fill_mode'],
#             type="string",
#             enum=["constant", "nearest", "reflect", "wrap"],
#             section="Preprocessing",
#         ),
#         "fill_mode_sub": Param(
#             config['preprocessing']['fill_mode_sub'],
#             type="string",
#             enum=["constant", "nearest", "reflect", "wrap"],
#             section="Preprocessing",
#         ),
#         "fill_value": Param(config['preprocessing']['fill_value'], type="integer", section="Preprocessing"),
#         "preprocessing_random_state": Param(config['preprocessing']['random_state'], type="integer", section="Preprocessing")
#     }

# def get_modeling_flat_params(config):
#     return {
#         "learning_rate": Param(config['modeling']['learning_rate'], type="number", minimum=0, section="Modeling"),
#         "center_scaled": Param(config['modeling']['center_scaled'], type="boolean", section="Modeling"),
#         "modeling_random_state": Param(config['modeling']['random_state'], type="integer", section="Modeling")
#     }

# def get_modeling_augmentation_params(config):
#     return {
#         "flip": Param(config['modeling']['augmentations']['flip'], type="boolean", section="Modeling - Augmentations"),
#         "brightness": Param(config['modeling']['augmentations']['brightness'], type="number", minimum=0, section="Modeling - Augmentations"),
#         "contrast": Param(config['modeling']['augmentations']['contrast'], type="number", minimum=0, section="Modeling - Augmentations"),
#         "saturation": Param(
#             config['modeling']['augmentations']['saturation'],
#             type="array",
#             minItems=2,
#             maxItems=2,
#             items={"type": "number", "minimum": 0, "maximum": 1},
#             section="Modeling - Augmentations",
#         ),
#         "hue": Param(config['modeling']['augmentations']['hue'], type="number", minimum=0, section="Modeling - Augmentations")
#     }

# def get_modeling_conv_blocks_params(config):
#     return {
#         "conv_blocks": Param(config['modeling']['conv_blocks'], type="array", items={
#             "type": "object",
#             "properties": {
#                 "filters": {"type": "integer", "minimum": 1},
#                 "dropout": {"type": "number", "minimum": 0, "maximum": 1},
#                 "normalization": {"type": "boolean"}
#             },
#             "required": ["filters"]
#         }, section="Modeling - Convolutional Blocks")
#     }

# def get_modeling_dense_blocks_params(config):
#     return {
#         "dense_blocks": Param(config['modeling']['dense_blocks'], type="array", items={
#             "type": "object",
#             "properties": {
#                 "units": {"type": "integer", "minimum": 1},
#                 "l2": {"type": "number", "minimum": 0},
#                 "dropout": {"type": "number", "minimum": 0, "maximum": 1},
#                 "normalization": {"type": "boolean"}
#             },
#             "required": ["units"]
#         }, section="Modeling - Dense Blocks")
#     }

# def get_modeling_params(config):
#     params = {}
#     params.update(get_modeling_flat_params(config))
#     params.update(get_modeling_augmentation_params(config))
#     params.update(get_modeling_conv_blocks_params(config))
#     params.update(get_modeling_dense_blocks_params(config))
#     return params

# def get_training_flat_params(config):
#     return {
#         "batch_size": Param(config['training']['batch_size'], type="integer", minimum=1, section="Training"),
#         "epochs": Param(config['training']['epochs'], type="integer", minimum=1, section="Training"),
#         "training_random_state": Param(config['training']['random_state'], type="integer", section="Training")
#     }

# def get_training_early_stopping_params(config):
#     return {
#         "early_stopping_min_delta": Param(config['training']['early_stopping']['min_delta'], type="number", minimum=0, section="Training - Early Stopping"),
#         "early_stopping_patience": Param(config['training']['early_stopping']['patience'], type="integer", minimum=0, section="Training - Early Stopping")
#     }

# def get_training_reduce_learning_rate_on_plateau_params(config):
#     return {
#         "plateau_min_delta": Param(config['training']['reduce_learning_rate_on_plateau']['min_delta'], type="number", minimum=0, section="Training - Reduce Learning Rate on Plateau"),
#         "plateau_patience": Param(config['training']['reduce_learning_rate_on_plateau']['patience'], type="integer", minimum=0, section="Training - Reduce Learning Rate on Plateau"),
#         "factor": Param(config['training']['reduce_learning_rate_on_plateau']['factor'], type="number", minimum=0, maximum=1, section="Training - Reduce Learning Rate on Plateau"),
#         "cooldown": Param(config['training']['reduce_learning_rate_on_plateau']['cooldown'], type="integer", minimum=0, section="Training - Reduce Learning Rate on Plateau")
#     }

# def get_training_params(config):
#     params = {}
#     params.update(get_training_flat_params(config))
#     params.update(get_training_early_stopping_params(config))
#     params.update(get_training_reduce_learning_rate_on_plateau_params(config))
#     return params

# def get_evaluation_params(config):
#     return {
#         "evaluation_threshold": Param(config['evaluation']['threshold'], type="number", minimum=0, maximum=1, section="Evaluation")
#     }

# def get_preparation_and_preprocessing_params(config):
#     params = {}
#     params.update(get_preparation_params(config))
#     params.update(get_preprocessing_params(config))
#     return params

# def get_modeling_and_training_params(config):
#     params = {}
#     params.update(get_modeling_params(config))
#     params.update(get_training_params(config))
#     return params

# def get_all_params(config):
#     params = {}
#     params.update(get_preparation_and_preprocessing_params(config))
#     params.update(get_modeling_and_training_params(config))
#     params.update(get_evaluation_params(config))
#     return params

# def update_config_with_preparation_and_preprocessing_params(config, params):
#     config['preparation']['train_test_split'] = params["train_test_split"]
#     config['preparation']['random_state'] = params["preparation_random_state"]
#     config['preprocessing']['patch_size'] = params["patch_size"]
#     config['preprocessing']['patches'] = params["patches"]
#     config['preprocessing']['overlap'] = params["overlap"]
#     config['preprocessing']['good_fraction'] = params["good_fraction"]
#     config['preprocessing']['oversampling'] = params["oversampling"]
#     config['preprocessing']['threshold'] = params["preprocessing_threshold"]
#     config['preprocessing']['threshold_factor'] = params["threshold_factor"]
#     config['preprocessing']['height_cropping'] = params["height_cropping"]
#     config['preprocessing']['width_cropping'] = params["width_cropping"]
#     config['preprocessing']['random_trans'] = params["random_trans"]
#     config['preprocessing']['random_rot'] = params["random_rot"]
#     config['preprocessing']['random_trans_sub'] = params["random_trans_sub"]
#     config['preprocessing']['random_rot_sub'] = params["random_rot_sub"]
#     config['preprocessing']['fill_mode'] = params["fill_mode"]
#     config['preprocessing']['fill_mode_sub'] = params["fill_mode_sub"]
#     config['preprocessing']['fill_value'] = params["fill_value"]
#     config['preprocessing']['random_state'] = params["preprocessing_random_state"]

# def update_config_with_modeling_and_training_params(config, params):
#     config['modeling']['learning_rate'] = params["learning_rate"]
#     config['modeling']['augmentations']['flip'] = params["flip"]
#     config['modeling']['augmentations']['brightness'] = params["brightness"]
#     config['modeling']['augmentations']['contrast'] = params["contrast"]
#     config['modeling']['augmentations']['saturation'] = params["saturation"]
#     config['modeling']['augmentations']['hue'] = params["hue"]
#     config['modeling']['center_scaled'] = params["center_scaled"]
#     config['modeling']['random_state'] = params["modeling_random_state"]
#     for i, conv_block in enumerate(config['modeling']['conv_blocks']):
#         conv_block['filters'] = params["conv_blocks"][i]["filters"]
#         conv_block['dropout'] = params["conv_blocks"][i]["dropout"]
#         conv_block['normalization'] = params["conv_blocks"][i]["normalization"]
#     for i, dense_block in enumerate(config['modeling']['dense_blocks']):
#         dense_block['units'] = params["dense_blocks"][i]["units"]
#         dense_block['l2'] = params["dense_blocks"][i]["l2"]
#         dense_block['dropout'] = params["dense_blocks"][i]["dropout"]
#         dense_block['normalization'] = params["dense_blocks"][i]["normalization"]
#     config['training']['batch_size'] = params["batch_size"]
#     config['training']['epochs'] = params["epochs"]
#     config['training']['random_state'] = params["training_random_state"]
#     config['training']['early_stopping']['min_delta'] = params["early_stopping_min_delta"]
#     config['training']['early_stopping']['patience'] = params["early_stopping_patience"]
#     config['training']['reduce_learning_rate_on_plateau']['min_delta'] = params["plateau_min_delta"]
#     config['training']['reduce_learning_rate_on_plateau']['patience'] = params["plateau_patience"]
#     config['training']['reduce_learning_rate_on_plateau']['factor'] = params["factor"]
#     config['training']['reduce_learning_rate_on_plateau']['cooldown'] = params["cooldown"]

# def update_config_with_evaluation_params(config, params):
#     config['evaluation']['threshold'] = params["evaluation_threshold"]

# def update_config_with_all_params(config, params):
#     update_config_with_preparation_and_preprocessing_params(config, params)
#     update_config_with_modeling_and_training_params(config, params)
#     update_config_with_evaluation_params(config, params)
