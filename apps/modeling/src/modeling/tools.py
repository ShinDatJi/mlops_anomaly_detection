def _extract_preparation_params(rep, params):
    for k, v in rep.items():
        params[f"preparation_{k}"] = v

def _extract_preprocessing_params(rep, params):
    for k, v in rep.items():
        params[f"preprocessing_{k}"] = v

def _extract_modeling_params(rep, params):
    for k, v in rep.items():
        params[f"modeling_{k}"] = v
    for k, v in rep["augmentations"].items():
        params[f"modeling_augmentations_{k}"] = v
    if type(rep["augmentations"]["saturation"]) is list:
        params["modeling_augmentations_saturation_min"] = rep["augmentations"]["saturation"][0]
        params["modeling_augmentations_saturation_max"] = rep["augmentations"]["saturation"][1]
    else:
        params["modeling_augmentations_saturation_min"] = 0
        params["modeling_augmentations_saturation_max"] = rep["augmentations"]["saturation"]
    del params["modeling_augmentations_saturation"]
    del params["modeling_augmentations"]
    def add_metrics_blocks(block_type):
        params[f"modeling_{block_type}"] = len(rep[block_type])
        for i in range(len(rep[block_type])):
            for k, v in rep[block_type][i].items():
                params[f"modeling_{block_type}_{i}_{k}"] = v
    add_metrics_blocks("conv_blocks")
    add_metrics_blocks("dense_blocks")

def _extract_training_params(rep, params):
    for k, v in rep.items():
        params[f"training_{k}"] = v
    for k, v in rep["early_stopping"].items():
        params[f"training_early_stopping_{k}"] = v
    del params["training_early_stopping"]
    for k, v in rep["reduce_learning_rate_on_plateau"].items():
        params[f"training_reduce_learning_rate_on_plateau_{k}"] = v
    del params["training_reduce_learning_rate_on_plateau"]

def _extract_evaluation_params(rep, params):
    for k, v in rep.items():
        params[f"evaluation_{k}"] = v

def extract_params_from_report(report):
    params = {}
    params["category"] = report["category"]
    params["img_size"] = report["img_size"]
    params["grayscale"] = report["grayscale"]

    if "preparation" in report:
        _extract_preparation_params(report["preparation"]["params"], params)
    if "preprocessing" in report:
        _extract_preprocessing_params(report["preprocessing"]["params"], params)
    if "modeling" in report:
        _extract_modeling_params(report["modeling"]["params"], params)
    if "training" in report:
        _extract_training_params(report["training"]["params"], params)
    if "evaluation" in report:
        _extract_evaluation_params(report["evaluation"]["params"], params)

    return params

def extract_preprocessing_metrics_from_report(report):
    rep = report["preprocessing"]["metrics"]
    metrics = rep.copy()
    def add_metrics(subset):
        for k, v in rep[subset].items():
            metrics[f"{subset}_{k}"] = v
        for k, v in rep[subset]["anomalies"].items():
            metrics[f"{subset}_anomalies_{k}"] = v
        del metrics[f"{subset}_anomalies"]
        del metrics[subset]
    add_metrics("train_images")
    add_metrics("test_images")
    add_metrics("test_patching_images")
    return metrics

def extract_training_metrics_from_report(report):
    rep = report["training"]["metrics"]
    metrics = rep.copy()
    for k, v in rep["scores"].items():
        metrics[f"scores_{k}"] = v
    del metrics["scores"]
    return metrics

def extract_evaluation_metrics_from_report(report):
    metrics = {}
    rep = report["evaluation"]["metrics"]
    def add_metrics(subset):
        for k, v in rep[subset].items():
            metrics[f"{subset}_{k}"] = v
    add_metrics("test_patching")
    add_metrics("test")
    add_metrics("train")
    return metrics