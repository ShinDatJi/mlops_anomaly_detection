import pandas as pd
import json
import os

keys = []
for d in os.scandir("./data/raw"):
    if d.is_dir():
        cat = d.name
        # src = os.path.join("../sept25int_bds_anomaly_detection/notebooks/reports/", cat, "final/report.json")
        # dst = os.path.join("./models/", cat, "report.json")
        # os.system(f"cp {src} {dst}")

        with open(os.path.join("./models", cat, "config.json"), "r") as f:
            config = json.load(f)

        part = "preprocessing"
        c = config[part]
        # print();print(cat)
        # print(len(c.keys()))
        # my_keys = []
        # count = 0
        # for k in c.keys():
        #     my_keys.append(k)
        #     if not k in keys:
        #         count += 1
        #         print(k)
        # keys = my_keys

        k = "threshold_factor"
        if k in c:
            print(c[k], cat)

        k_prev = "threshold"
        v = 1
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