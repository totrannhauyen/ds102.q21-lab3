
import os
import cv2
import numpy as np

def load_data(data_dir, img_size=128):

    X, y = [], []
    total = 0
    skipped = 0

    for folder in ["NORMAL", "PNEUMONIA"]:
        path = os.path.join(data_dir, folder)

        if not os.path.exists(path):
            print(f"Không tìm thấy folder: {path}")
            continue

        for file in os.listdir(path):
            img_path = os.path.join(path, file)
            total += 1

            if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                skipped += 1
                continue

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None or img.size == 0:
                print(f"Lỗi ảnh: {img_path}")
                skipped += 1
                continue

            try:
                img = cv2.resize(img, (img_size, img_size))
            except:
                print(f"Resize lỗi: {img_path}")
                skipped += 1
                continue

            img = img / 255.0

            X.append(img.flatten())
            y.append(-1 if folder == "NORMAL" else 1)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int8)

    print(f"\nLoaded: {len(X)} ảnh")
    print(f"Skipped: {skipped} / {total}")

    return X, y
