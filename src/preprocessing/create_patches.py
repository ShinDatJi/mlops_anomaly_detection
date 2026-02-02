from pathlib import Path
import os
import cv2
import numpy as np
import keras.layers as layers

def prepare_folders(data_dir, subpath, category):
    path = data_dir + category + "/" + subpath + "/"
    Path(path + "0/").mkdir(parents=True, exist_ok=True)
    Path(path + "1/").mkdir(parents=True, exist_ok=True)
    for f in os.listdir(path + "0/"):
        os.remove(path + "0/" + f)
    for f in os.listdir(path + "1/"):
        os.remove(path + "1/" + f)
    return path

def create_images(data_dir, df, subpath, img_size, oversampling, random_trans=0, random_rot=0, fill_mode="reflect", fill_value=0, seed=None):

    path = prepare_folders(data_dir, subpath, df.category.iloc[0])

    samples = int((df.anomaly == "good").sum() / (df.anomaly != "good").sum()) if oversampling else 1

    # trans = layers.RandomTranslation(random_trans, random_trans, fill_mode=fill_mode, fill_value=fill_value, seed=seed)
    # rot = layers.RandomRotation(random_rot, fill_mode=fill_mode, fill_value=fill_value, seed=seed)

    print(subpath)
    count_1 = 0
    count_0 = 0
    for i in df.index:
        img = cv2.imread(df.loc[i].file)
        # if random_trans:
        #     img = trans.call(img)
        # if random_rot:
        #     img = rot.call(img)
        # if random_rot or random_trans:
        #     img = img.numpy()
        img = cv2.resize(img, dsize=(img_size, img_size))
        if df.loc[i].anomaly == "good":
            cv2.imwrite(path + "0/" + str(i).zfill(4) + ".png", img)
            count_0 += 1
        else:
            for s in range(samples):
                cv2.imwrite(path + "1/" + str(i).zfill(4) + "_" + str(s).zfill(3) + ".png", img)
                count_1 += 1
    images_0 = len(df[df.anomaly == "good"])
    images_1 = len(df[df.anomaly != "good"])
    print(  "0 images:", count_0)
    print(  "1 images:", count_1)
    if oversampling:
        print(  "1 oversampling:", samples - 1)

    return {
        "images_good_source": images_0,
        "images_bad_source": images_1,
        "oversampling": samples - 1,
        "images_good": count_0,
        "images_bad": count_1,
    }

def create_patches(
        data_dir, df, subpath, patch_size, patches, bw=False, good_fraction=False,
        oversampling=False, keep_all=False, keep_good=False, spread=0.1, threshold="auto", threshold_factor=1, overlap=0,
        height_cropping=0, width_cropping=0, embedding=False, fast_patching=True,
        random_trans=0, random_rot=0, random_trans_sub=0, random_rot_sub=0,
        fill_mode="reflect", fill_mode_sub="reflect", fill_value=0, seed=None):

    # if good_fraction:
    #     df_0 = df[df.anomaly == "good"].sample(frac=good_fraction, random_state=seed)
    #     df_1 = df[df.anomaly != "good"]
    #     df = pd.concat([df_0, df_1], axis=0)

    path = prepare_folders(data_dir, subpath, df.category.iloc[0])
    img_size = df.img_size.iloc[0]

    step = int(img_size / (patches + overlap / (1 - overlap)))
    patch_size_overlap = int(step / (1 - overlap))
    new_img_size = (patches - 1) * step + patch_size_overlap
    start = (img_size - new_img_size) // 2
    # end = new_img_size - patch_size_overlap + start + 1
    # iter_range = range(start, end, step)

    trans = layers.RandomTranslation(random_trans, random_trans, fill_mode=fill_mode, fill_value=fill_value, seed=seed)
    rot = layers.RandomRotation(random_rot, fill_mode=fill_mode, fill_value=fill_value, seed=seed)
    trans_sub = layers.RandomTranslation(random_trans_sub, random_trans_sub, fill_mode=fill_mode_sub, fill_value=fill_value, seed=seed)
    rot_sub = layers.RandomRotation(random_rot_sub, fill_mode=fill_mode_sub, fill_value=fill_value, seed=seed)
    
    rng = np.random.default_rng(seed)

    print(subpath)
    count_0 = 0
    for i in df[df.anomaly == "good"].index:
        img = cv2.imread(df.loc[i].file)
        if embedding:
            img_small = cv2.resize(img, dsize=(embedding, embedding))
        if random_trans:
            img = trans(img)
        if random_rot:
            img = rot(img)
        if random_rot or random_trans:
            img = img.numpy()
        for p_r in range(height_cropping, patches - height_cropping):
            r = start + p_r * step
            for p_c in range(width_cropping, patches - width_cropping):
                c = start + p_c * step
        # for r, c in itertools.product(iter_range, iter_range):
                if rng.random() >= good_fraction:
                    continue
                patch = img[r:r + patch_size_overlap, c:c + patch_size_overlap, :]
                patch = cv2.resize(patch, dsize=(patch_size, patch_size))
                if random_trans_sub:
                    patch = trans_sub(patch)
                if random_rot_sub:
                    patch = rot_sub(patch)
                if random_rot_sub or random_trans_sub:
                    patch = patch.numpy()
                if embedding:
                    patch[0:embedding, 0:embedding] = img_small
                cv2.imwrite(path + "0/" + str(i).zfill(4) + "_" + str(r).zfill(4) + "_" + str(c).zfill(4) + ".png", patch)
                count_0 += 1
    print("  0 patches:", count_0)

    seed *= 2
    trans = layers.RandomTranslation(random_trans, random_trans, fill_mode=fill_mode, fill_value=fill_value, seed=seed)
    rot = layers.RandomRotation(random_rot, fill_mode=fill_mode, fill_value=fill_value, seed=seed)
    trans_mask = layers.RandomTranslation(random_trans, random_trans, fill_mode=fill_mode, fill_value=0, seed=seed)
    rot_mask = layers.RandomRotation(random_rot, fill_mode=fill_mode, fill_value=0, seed=seed)
    trans_sub = layers.RandomTranslation(random_trans_sub, random_trans_sub, fill_mode=fill_mode_sub, fill_value=fill_value, seed=seed)
    rot_sub = layers.RandomRotation(random_rot_sub, fill_mode=fill_mode_sub, fill_value=fill_value, seed=seed)
    trans_sub_mask = layers.RandomTranslation(random_trans_sub, random_trans_sub, fill_mode=fill_mode_sub, fill_value=0, seed=seed)
    rot_sub_mask = layers.RandomRotation(random_rot_sub, fill_mode=fill_mode_sub, fill_value=0, seed=seed)

    if threshold == "auto":
        threshold = df[df.anomaly != "good"].anomaly_coverage.min() / spread
    if threshold == "full-auto":
        threshold = df[df.anomaly != "good"].anomaly_coverage.min() * img_size**2 / patch_size_overlap**2
        threshold *= threshold_factor
    if keep_all:
        threshold = 0
    print(f"  1 threshold: {threshold:0.3f}")

    # images = {}
    # masks = {}
    # for i in df[df.anomaly != "good"].index:
    #     images[i] = cv2.imread(df.loc[i].file)
    #     mask = cv2.imread(df.loc[i].file_ground_truth)
    #     mask = mask.astype(float) / 255
    #     mask = mask[:,:,:1]
    #     masks[i] = mask

    stats = {}
    for c in df[df.anomaly != "good"].anomaly.values:
        stats[c] = 0

    count_1 = 0
    count = 0
    while count_1 <= count_0:
    # while count_1 < count_0 - len(df[df.anomaly != "good"]):
        for i in df[df.anomaly != "good"].index:
            img = cv2.imread(df.loc[i].file)
            mask = cv2.imread(df.loc[i].file_ground_truth)
            mask = mask.astype(float) / 255
            mask = mask[:,:,:1]
            if embedding:
                img_small = cv2.resize(img, dsize=(embedding, embedding))
            if random_trans:
                img = trans(img)
                mask = trans_mask(mask)
            if random_rot:
                img = rot(img)
                mask = rot_mask(mask)
            if random_trans or random_rot:
                img = img.numpy()
                mask = mask.numpy()
            if fast_patching and not(keep_all or (keep_good and not embedding and count == 0)) and mask.mean() == 0:
                continue
            # for r, c in itertools.product(iter_range, iter_range):
            for p_r in range(height_cropping, patches - height_cropping):
                r = start + p_r * step
                for p_c in range(width_cropping, patches - width_cropping):
                    c = start + p_c * step
                    patch = img[r:r + patch_size_overlap, c:c + patch_size_overlap, :]
                    patch.resize()
                    patch = cv2.resize(patch, dsize=(patch_size, patch_size))
                    mask_patch = mask[r:r + patch_size_overlap, c:c + patch_size_overlap, :]
                    if fast_patching and not(keep_all or (keep_good and not embedding and count == 0)) and mask_patch.mean() == 0:
                        continue
                    mask_patch = cv2.resize(mask_patch, dsize=(patch_size, patch_size))
                    mask_patch = mask_patch[..., np.newaxis]
                    if random_trans_sub:
                        patch = trans_sub(patch)
                        mask_patch = trans_sub_mask(mask_patch)
                    if random_rot_sub:
                        patch = rot_sub(patch)
                        mask_patch = rot_sub_mask(mask_patch)
                    if random_trans_sub or random_rot_sub:
                        patch = patch.numpy()
                        mask_patch = mask_patch.numpy()
                    if embedding:
                        patch[0:embedding, 0:embedding] = img_small
                        mask_patch[0:embedding, 0:embedding] = np.zeros((embedding, embedding))
                        # cv2.imwrite(path + str(i).zfill(4) + "_" + str(count).zfill(3) + "_" + str(r).zfill(4) + "_" + str(c).zfill(4) + ".png", mask_patch * 255)
                    if mask_patch.mean() >= threshold: 
                        cv2.imwrite(path + "1/" + str(i).zfill(4) + "_" + str(count).zfill(3) + "_" + str(r).zfill(4) + "_" + str(c).zfill(4) + ".png", patch)
                        count_1 += 1
                        stats[df.loc[i].anomaly] += 1
                    elif keep_good and not embedding and count == 0 and mask_patch.mean() == 0:
                        cv2.imwrite(path + "0/" + str(i).zfill(4) + "_" + str(r).zfill(4) + "_" + str(c).zfill(4) + ".png", patch)
                        count_0 += 1
        if keep_good and count == 0:
            print("  0 patches:", count_0)
        print("  1 patches:", count_1, end="\r")
        if not oversampling:
            break
        count += 1
    print()
    if oversampling:
        print("  1 oversampling:", count)

    print(stats)
            
    return {
        "images_good_source": len(df[df.anomaly == "good"]),
        "images_bad_source": len(df[df.anomaly != "good"]),
        "oversampling": count,
        "images_good": count_0,
        "images_bad": count_1,
    }, threshold
