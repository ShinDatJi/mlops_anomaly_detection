import pandas as pd
import os

df = pd.read_csv("./data/mad.csv")
for c in df.category.unique():
    print(c)
    os.makedirs("./models/" + c, exist_ok=True)
    src = "./notebooks/reports/" + c + "/final/config.json"
    dst = "./models/" + c + "/config.json"
    os.system("cp {src} {dst}".format(src=src, dst=dst))
    