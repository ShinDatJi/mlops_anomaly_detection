import pandas as pd
import json
import os

def func1():
    keys = []
    for d in os.scandir("./data/raw"):
        if d.is_dir():
            cat = d.name
            # src = os.path.join("../sept25int_bds_anomaly_detection/notebooks/reports/", cat, "final/report.json")
            # dst = os.path.join("./models/", cat, "report.json")
            # os.system(f"cp {src} {dst}")

            with open(os.path.join("./models", cat, "config.json"), "r") as f:
                config = json.load(f)

            part = "training"
            c = config[part]

            k = "use_buggy_early_stopping_restore_best_weights"
            if k in c:
                print(c[k], cat)

            k_prev = "epochs"
            v = False
            if not k in c:
                keys = list(c.keys())
                values = list(c.values())
                tar_idx = keys.index(k_prev) # index of the target key 'b'
                a = keys[:tar_idx + 1] + [k] + keys[tar_idx + 1:]
                b = values[:tar_idx + 1] + [v] + values[tar_idx + 1:]
                res = dict(zip(a, b))
                config[part] = res
            with open(os.path.join("./models", cat, "config.json"), "w") as f:
                json.dump(config, f, indent=2)

def func2():
    for d in os.scandir("./data/raw"):
        if d.is_dir():
            cat = d.name
            with open(os.path.join("./models", cat, "report.json"), "r") as f:
                report = json.load(f)

            rep = report["evaluation"]
            params = rep["test_patching"]["params"]
            metrics_test = rep["test"]["metrics"]
            metrics_train = rep["train"]["metrics"]
            metrics_test_patching = params.copy()
            metrics_test_patching.update(rep["test_patching"]["metrics"])
            report["evaluation"] = {
                "params": params,
                "metrics": {
                    "test": metrics_test,
                    "train": metrics_train,
                    "test_patching": metrics_test_patching
                }
            }

            with open(os.path.join("./models", cat, "report.json"), "w") as f:
                json.dump(report, f, indent=2)

def func3():
    for d in os.scandir("./data/raw"):
        if d.is_dir():
            cat = d.name
            src = f"./reports/modeling/{cat}/evaluation_report.json"
            dst = f"./models/{cat}/report.json"
            os.system(f"cp {src} {dst}")

func2()
