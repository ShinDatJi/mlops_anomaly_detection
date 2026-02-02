import pandas as pd
import json

df = pd.read_csv("./data/mad.csv")

for c in df.category.unique():
    with open("./models/" + c + "/config.json", "r") as f:
        config = json.load(f)
    del config["patching"]
    del config["preprocessing"]["embedding"]
    del config["training"]["class_weight"]
    config["validation"] = config["testing"]
    del config["testing"]
    with open("./models/" + c + "/config.json", "w") as f:
        json.dump(config, f, indent=2)
