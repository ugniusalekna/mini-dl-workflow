import os
import random
import numpy as np
from PIL import Image


def get_image_paths(root_dir, extensions=("jpg", "jpeg", "png")):
    image_paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths


def read_image(img_path):
    image = Image.open(img_path).convert("RGB")
    return np.array(image)


def get_cls_from_path(img_path):
    return os.path.basename(os.path.dirname(img_path))


def make_class_map(root_dir):
    return {cls_name: idx for idx, cls_name in enumerate(sorted(os.listdir(root_dir)))}


def reverse_class_map(class_map):
    return {v: k for k, v in class_map.items()}


def train_val_split(img_paths, val_ratio=0.2, seed=None):
    if seed is not None:
        old_state = random.getstate()
        random.seed(seed)
    random.shuffle(img_paths)
    if seed is not None:
        random.setstate(old_state)

    split = int(len(img_paths) * (1 - val_ratio))
    return img_paths[:split], img_paths[split:]