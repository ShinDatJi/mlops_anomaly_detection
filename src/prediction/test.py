import os
import predict

def predict_all_test(category = None):
    for d in os.scandir("./data/raw"):
        if d.is_dir():
            cat = d.name
            if category and category != cat:
                continue
            print();print(cat)
            count_p = 0
            count_p_f = 0
            count_n = 0
            count_n_f = 0
            for a in os.scandir(os.path.join(d.path, "test")):
                for i in os.listdir(a.path):
                    image_path = os.path.join(a.path, i)
                    with open(image_path, "rb") as f:
                        image_bin = f.read()
                    pred = predict.predict(cat, image_bin)
                    real = int(a.name != "good")
                    if real != pred:
                        print(real == pred, real, pred, image_path)
                        if(real):
                            count_n_f += 1
                        else:
                            count_p_f += 1
                    # print(pred_probas)
                    if real:
                        count_p += 1
                    else:
                        count_n += 1
            print("tpr:", (count_p - count_n_f), "/", count_p, "tnr:", (count_n - count_p_f), "/", count_n)

def predict_one(category):
    for d in os.scandir("./data/raw"):
        if d.is_dir():
            cat = d.name
            if category and category != cat:
                continue
            print(cat)
            for a in os.scandir(os.path.join(d.path, "test")):
                i = os.listdir(a.path)[0]
                image_path = os.path.join(a.path, i)
                with open(image_path, "rb") as f:
                    image_bin = f.read()
                pred = predict.predict(cat, image_bin)
                print(pred)
                break;

# predict_all_test()
predict_one("capsule")