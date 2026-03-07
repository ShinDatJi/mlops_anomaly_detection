import os
import cv2
import numpy as np
import keras.layers as layers

def prepare_folders(data_path):
    path0 = os.path.join(data_path, "0")
    path1 = os.path.join(data_path, "1")
    os.makedirs(path0, exist_ok=True)
    os.makedirs(path1, exist_ok=True)
    for e in os.scandir(path0):
        os.remove(e.path)
    for e in os.scandir(path1):
        os.remove(e.path)
    return path0, path1

def create_patches(
        data_path, df, img_size, random_state, patch_size, patches, keep_all=False, keep_good=False, 
        good_fraction=1, spread=0.1, threshold_mode="full-auto", threshold=0.01, threshold_factor=1, overlap=0,
        height_cropping=0, width_cropping=0, fast_patching=True, oversampling=True,
        random_trans=0, random_rot=0, random_trans_sub=0, random_rot_sub=0,
        fill_mode="constant", fill_mode_sub="constant", fill_value=0):

    if threshold_mode != "use_threshold":
        if threshold_mode == "auto":
            threshold = "full_auto"

    path0, path1 = prepare_folders(data_path)

    step = int(img_size / (patches + overlap / (1 - overlap)))
    patch_size_overlap = int(step / (1 - overlap))
    new_img_size = (patches - 1) * step + patch_size_overlap
    start = (img_size - new_img_size) // 2

    trans = layers.RandomTranslation(random_trans, random_trans, fill_mode=fill_mode, fill_value=fill_value, seed=random_state)
    rot = layers.RandomRotation(random_rot, fill_mode=fill_mode, fill_value=fill_value, seed=random_state)
    trans_sub = layers.RandomTranslation(random_trans_sub, random_trans_sub, fill_mode=fill_mode_sub, fill_value=fill_value, seed=random_state)
    rot_sub = layers.RandomRotation(random_rot_sub, fill_mode=fill_mode_sub, fill_value=fill_value, seed=random_state)
    
    rng = np.random.default_rng(random_state)

    print(data_path)
    count_0 = 0
    for i in df[df.anomaly == "good"].index:
        img = cv2.imread(df.loc[i].file)
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
                file = f"{str(i).zfill(4)}_{str(r).zfill(4)}_{str(c).zfill(4)}.png"
                cv2.imwrite(os.path.join(path0, file), patch)
                count_0 += 1
    print("  0 patches:", count_0)

    random_state *= 2
    trans = layers.RandomTranslation(random_trans, random_trans, fill_mode=fill_mode, fill_value=fill_value, seed=random_state)
    rot = layers.RandomRotation(random_rot, fill_mode=fill_mode, fill_value=fill_value, seed=random_state)
    trans_mask = layers.RandomTranslation(random_trans, random_trans, fill_mode=fill_mode, fill_value=0, seed=random_state)
    rot_mask = layers.RandomRotation(random_rot, fill_mode=fill_mode, fill_value=0, seed=random_state)
    trans_sub = layers.RandomTranslation(random_trans_sub, random_trans_sub, fill_mode=fill_mode_sub, fill_value=fill_value, seed=random_state)
    rot_sub = layers.RandomRotation(random_rot_sub, fill_mode=fill_mode_sub, fill_value=fill_value, seed=random_state)
    trans_sub_mask = layers.RandomTranslation(random_trans_sub, random_trans_sub, fill_mode=fill_mode_sub, fill_value=0, seed=random_state)
    rot_sub_mask = layers.RandomRotation(random_rot_sub, fill_mode=fill_mode_sub, fill_value=0, seed=random_state)

    if threshold == "auto":
        threshold = df[df.anomaly != "good"].anomaly_coverage.min() / spread
    if threshold == "full-auto":
        threshold = df[df.anomaly != "good"].anomaly_coverage.min() * img_size**2 / patch_size_overlap**2
        threshold *= threshold_factor
    if keep_all:
        threshold = 0
    print(f"  1 threshold: {threshold:0.3f}")

    stats = {}
    for c in df[df.anomaly != "good"].anomaly.values:
        stats[c] = 0

    count_1 = 0
    count = 0
    while count_1 <= count_0:
        for i in df[df.anomaly != "good"].index:
            img = cv2.imread(df.loc[i].file)
            mask = cv2.imread(df.loc[i].file_ground_truth)
            mask = mask.astype(float) / 255
            mask = mask[:,:,:1]
            if random_trans:
                img = trans(img)
                mask = trans_mask(mask)
            if random_rot:
                img = rot(img)
                mask = rot_mask(mask)
            if random_trans or random_rot:
                img = img.numpy()
                mask = mask.numpy()
            if fast_patching and not(keep_all or (keep_good and count == 0)) and mask.mean() == 0:
                continue
            for p_r in range(height_cropping, patches - height_cropping):
                r = start + p_r * step
                for p_c in range(width_cropping, patches - width_cropping):
                    c = start + p_c * step
                    patch = img[r:r + patch_size_overlap, c:c + patch_size_overlap, :]
                    patch.resize()
                    patch = cv2.resize(patch, dsize=(patch_size, patch_size))
                    mask_patch = mask[r:r + patch_size_overlap, c:c + patch_size_overlap, :]
                    if fast_patching and not(keep_all or (keep_good and count == 0)) and mask_patch.mean() == 0:
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
                    if mask_patch.mean() >= threshold: 
                        file = f"{str(i).zfill(4)}_{str(count).zfill(3)}_{str(r).zfill(4)}_{str(c).zfill(4)}.png"
                        cv2.imwrite(os.path.join(path1, file), patch)
                        count_1 += 1
                        stats[df.loc[i].anomaly] += 1
                    elif keep_good and count == 0 and mask_patch.mean() == 0:
                        file = f"{str(i).zfill(4)}_{str(r).zfill(4)}_{str(c).zfill(4)}.png"
                        cv2.imwrite(os.path.join(path0, file), patch)
                        count_0 += 1
        if keep_good and count == 0:
            print("  0 patches:", count_0)
        # print("  1 patches:", count_1, end="\r")
        print("  1 patches:", count_1)
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
    }, threshold, stats
