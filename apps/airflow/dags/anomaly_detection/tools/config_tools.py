import os
import json
from airflow.models import Param

default_config_file = os.environ["MODELING_DEFAULT_CONFIG_FILE"]
config_path = os.environ["MODELING_CONFIG_PATH"]
config_name = os.environ["MODELING_CONFIG"]

def load_default_config():
    with open(default_config_file, "r") as f:
        config = json.load(f)
    return config

def load_config(category, version):
    config_file = os.path.join(config_path, f"{category}_{version}", config_name)
    with open(config_file, "r") as f:
        config = json.load(f)
    return config

def save_config(category, version, config):
    config_file = os.path.join(config_path, f"{category}_{version}", config_name)
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

def get_preparation_params(config):
    return {
        "preparation_train_test_split": Param(config['preparation']['train_test_split'], type="number", minimum=0, maximum=1, section="Preparation"),
        "preparation_random_state": Param(config['preparation']['random_state'], type="integer", section="Preparation")
    }

def get_preprocessing_params(config):
    return {
        "preprocessing_patch_size": Param(config['preprocessing']['patch_size'], type="integer", minimum=1, section="Preprocessing"),
        "preprocessing_patches": Param(config['preprocessing']['patches'], type="integer", minimum=1, section="Preprocessing"),
        "preprocessing_overlap": Param(config['preprocessing']['overlap'], type="number", minimum=0, maximum=1, section="Preprocessing"),
        "preprocessing_good_fraction": Param(config['preprocessing']['good_fraction'], type="number", minimum=0, maximum=1, section="Preprocessing"),
        "preprocessing_oversampling": Param(config['preprocessing']['oversampling'], type="boolean", section="Preprocessing"),
        "preprocessing_threshold_mode": Param(
            config['preprocessing']['threshold_mode'],
            type="string",
            enum=["auto", "manual"],
            section="Preprocessing"
        ),
        "preprocessing_threshold": Param(config['preprocessing']['threshold'], type="number", minimum=0, maximum=1, section="Preprocessing"),
        "preprocessing_threshold_factor": Param(config['preprocessing']['threshold_factor'], type="number", minimum=0, section="Preprocessing"),
        "preprocessing_height_cropping": Param(config['preprocessing']['height_cropping'], type="integer", minimum=0, section="Preprocessing"),
        "preprocessing_width_cropping": Param(config['preprocessing']['width_cropping'], type="integer", minimum=0, section="Preprocessing"),
        "preprocessing_random_trans": Param(config['preprocessing']['random_trans'], type="number", minimum=0, maximum=1, section="Preprocessing"),
        "preprocessing_random_rot": Param(config['preprocessing']['random_rot'], type="number", minimum=0, maximum=1, section="Preprocessing"),
        "preprocessing_random_trans_sub": Param(config['preprocessing']['random_trans_sub'], type="number", minimum=0, maximum=1, section="Preprocessing"),
        "preprocessing_random_rot_sub": Param(config['preprocessing']['random_rot_sub'], type="number", minimum=0, maximum=1, section="Preprocessing"),
        "preprocessing_fill_mode": Param(
            config['preprocessing']['fill_mode'],
            type="string",
            enum=["constant", "nearest", "reflect", "wrap"],
            section="Preprocessing",
        ),
        "preprocessing_fill_mode_sub": Param(
            config['preprocessing']['fill_mode_sub'],
            type="string",
            enum=["constant", "nearest", "reflect", "wrap"],
            section="Preprocessing",
        ),
        "preprocessing_fill_value": Param(config['preprocessing']['fill_value'], type="integer", section="Preprocessing"),
        "preprocessing_random_state": Param(config['preprocessing']['random_state'], type="integer", section="Preprocessing")
    }

def get_modeling_flat_params(config):
    return {
        "modeling_learning_rate": Param(config['modeling']['learning_rate'], type="number", minimum=0, section="Modeling"),
        "modeling_center_scaled": Param(config['modeling']['center_scaled'], type="boolean", section="Modeling"),
        "modeling_random_state": Param(config['modeling']['random_state'], type="integer", section="Modeling")
    }

def get_modeling_augmentation_params(config):
    return {
        "modeling_augmentations_flip": Param(config['modeling']['augmentations']['flip'], type="boolean", section="Modeling - Augmentations"),
        "modeling_augmentations_brightness": Param(config['modeling']['augmentations']['brightness'], type="number", minimum=0, section="Modeling - Augmentations"),
        "modeling_augmentations_contrast": Param(config['modeling']['augmentations']['contrast'], type="number", minimum=0, section="Modeling - Augmentations"),
        "modeling_augmentations_saturation_min": Param(config['modeling']['augmentations']['saturation'][0], type="number", minimum=0, maximum=1, section="Modeling - Augmentations"),
        "modeling_augmentations_saturation_max": Param(config['modeling']['augmentations']['saturation'][1], type="number", minimum=0, maximum=1, section="Modeling - Augmentations"),
        "modeling_augmentations_hue": Param(config['modeling']['augmentations']['hue'], type="number", minimum=0, section="Modeling - Augmentations")
    }

def get_modeling_conv_blocks_params(config):
    return {
        "modeling_conv_blocks": Param(config['modeling']['conv_blocks'], type="array", items={
            "type": "object",
            "properties": {
                "filters": {"type": "integer", "minimum": 1},
                "dropout": {"type": "number", "minimum": 0, "maximum": 1},
                "normalization": {"type": "boolean"}
            },
            "required": ["filters"]
        }, section="Modeling - Convolutional Blocks")
    }

def get_modeling_dense_blocks_params(config):
    return {
        "modeling_dense_blocks": Param(config['modeling']['dense_blocks'], type="array", items={
            "type": "object",
            "properties": {
                "units": {"type": "integer", "minimum": 1},
                "l2": {"type": "number", "minimum": 0},
                "dropout": {"type": "number", "minimum": 0, "maximum": 1},
                "normalization": {"type": "boolean"}
            },
            "required": ["units"]
        }, section="Modeling - Dense Blocks")
    }

def get_modeling_params(config):
    params = {}
    params.update(get_modeling_flat_params(config))
    params.update(get_modeling_augmentation_params(config))
    params.update(get_modeling_conv_blocks_params(config))
    params.update(get_modeling_dense_blocks_params(config))
    return params

def get_training_flat_params(config):
    return {
        "training_batch_size": Param(config['training']['batch_size'], type="integer", minimum=1, section="Training"),
        "training_epochs": Param(config['training']['epochs'], type="integer", minimum=1, section="Training"),
        "training_random_state": Param(config['training']['random_state'], type="integer", section="Training")
    }

def get_training_early_stopping_params(config):
    return {
        "training_early_stopping_min_delta": Param(config['training']['early_stopping']['min_delta'], type="number", minimum=0, section="Training - Early Stopping"),
        "training_early_stopping_patience": Param(config['training']['early_stopping']['patience'], type="integer", minimum=0, section="Training - Early Stopping")
    }

def get_training_reduce_learning_rate_on_plateau_params(config):
    return {
        "training_reduce_learning_rate_on_plateau_min_delta": Param(config['training']['reduce_learning_rate_on_plateau']['min_delta'], type="number", minimum=0, section="Training - Reduce Learning Rate on Plateau"),
        "training_reduce_learning_rate_on_plateau_patience": Param(config['training']['reduce_learning_rate_on_plateau']['patience'], type="integer", minimum=0, section="Training - Reduce Learning Rate on Plateau"),
        "training_reduce_learning_rate_on_plateau_factor": Param(config['training']['reduce_learning_rate_on_plateau']['factor'], type="number", minimum=0, maximum=1, section="Training - Reduce Learning Rate on Plateau"),
        "training_reduce_learning_rate_on_plateau_cooldown": Param(config['training']['reduce_learning_rate_on_plateau']['cooldown'], type="integer", minimum=0, section="Training - Reduce Learning Rate on Plateau")
    }

def get_training_params(config):
    params = {}
    params.update(get_training_flat_params(config))
    params.update(get_training_early_stopping_params(config))
    params.update(get_training_reduce_learning_rate_on_plateau_params(config))
    return params

def get_evaluation_params(config):  
    return {
        "evaluation_threshold": Param(config['evaluation']['threshold'], type="number", minimum=0, maximum=1, section="Evaluation")
    }

def get_preparation_and_preprocessing_params(config):
    params = {}
    params.update(get_preparation_params(config))
    params.update(get_preprocessing_params(config))
    return params

def get_modeling_and_training_params(config):
    params = {}
    params.update(get_modeling_params(config))
    params.update(get_training_params(config))
    return params

def get_all_params(config):
    params = {}
    params.update(get_preparation_and_preprocessing_params(config))
    params.update(get_modeling_and_training_params(config))
    params.update(get_evaluation_params(config))
    return params

def update_config_with_preparation_and_preprocessing_params(config, params):
    config['preparation'] = {}
    config['preparation']['train_test_split'] = params["preparation_train_test_split"]
    config['preparation']['random_state'] = params["preparation_random_state"]
    config['preprocessing'] = {}
    config['preprocessing']['patch_size'] = params["preprocessing_patch_size"]
    config['preprocessing']['patches'] = params["preprocessing_patches"]
    config['preprocessing']['overlap'] = params["preprocessing_overlap"]
    config['preprocessing']['good_fraction'] = params["preprocessing_good_fraction"]
    config['preprocessing']['oversampling'] = params["preprocessing_oversampling"]
    config['preprocessing']['threshold_mode'] = params["preprocessing_threshold_mode"]
    config['preprocessing']['threshold'] = params["preprocessing_threshold"]
    config['preprocessing']['threshold_factor'] = params["preprocessing_threshold_factor"]
    config['preprocessing']['height_cropping'] = params["preprocessing_height_cropping"]
    config['preprocessing']['width_cropping'] = params["preprocessing_width_cropping"]
    config['preprocessing']['random_trans'] = params["preprocessing_random_trans"]
    config['preprocessing']['random_rot'] = params["preprocessing_random_rot"]
    config['preprocessing']['random_trans_sub'] = params["preprocessing_random_trans_sub"]
    config['preprocessing']['random_rot_sub'] = params["preprocessing_random_rot_sub"]
    config['preprocessing']['fill_mode'] = params["preprocessing_fill_mode"]
    config['preprocessing']['fill_mode_sub'] = params["preprocessing_fill_mode_sub"]
    config['preprocessing']['fill_value'] = params["preprocessing_fill_value"]
    config['preprocessing']['random_state'] = params["preprocessing_random_state"]

def update_config_with_modeling_and_training_params(config, params):
    config['modeling'] = {}
    config['modeling']['learning_rate'] = params["modeling_learning_rate"]
    config['modeling']['augmentations'] = {}
    config['modeling']['augmentations']['flip'] = params["modeling_augmentations_flip"]
    config['modeling']['augmentations']['brightness'] = params["modeling_augmentations_brightness"]
    config['modeling']['augmentations']['contrast'] = params["modeling_augmentations_contrast"]
    config['modeling']['augmentations']['saturation'] = [
        params["modeling_augmentations_saturation_min"],
        params["modeling_augmentations_saturation_max"]
    ]
    config['modeling']['augmentations']['hue'] = params["modeling_augmentations_hue"]
    config['modeling']['center_scaled'] = params["modeling_center_scaled"]
    config['modeling']['conv_blocks'] = params["modeling_conv_blocks"]
    config['modeling']['dense_blocks'] = params["modeling_dense_blocks"]
    config['modeling']['random_state'] = params["modeling_random_state"]
    config['training'] = {}
    config['training']['batch_size'] = params["training_batch_size"]
    config['training']['epochs'] = params["training_epochs"]
    config['training']['early_stopping'] = {}
    config['training']['early_stopping']['min_delta'] = params["training_early_stopping_min_delta"]
    config['training']['early_stopping']['patience'] = params["training_early_stopping_patience"]
    config['training']['reduce_learning_rate_on_plateau'] = {}
    config['training']['reduce_learning_rate_on_plateau']['min_delta'] = params["training_reduce_learning_rate_on_plateau_min_delta"]
    config['training']['reduce_learning_rate_on_plateau']['patience'] = params["training_reduce_learning_rate_on_plateau_patience"]
    config['training']['reduce_learning_rate_on_plateau']['factor'] = params["training_reduce_learning_rate_on_plateau_factor"]
    config['training']['reduce_learning_rate_on_plateau']['cooldown'] = params["training_reduce_learning_rate_on_plateau_cooldown"]
    config['training']['random_state'] = params["training_random_state"]

def update_config_with_evaluation_params(config, params):
    config['evaluation'] = {}
    config['evaluation']['threshold'] = params["evaluation_threshold"]

def update_config_with_all_params(config, params):
    update_config_with_preparation_and_preprocessing_params(config, params)
    update_config_with_modeling_and_training_params(config, params)
    update_config_with_evaluation_params(config, params)
