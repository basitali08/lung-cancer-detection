import os, sys
from datasets import load_dataset
from PIL import Image
import random

random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

LABEL_MAP = {
    0: "adenocarcinoma",
    1: "adenocarcinoma",
    2: "large_cell_carcinoma",
    3: "large_cell_carcinoma",
    4: "normal",
    5: "squamous_cell_carcinoma",
    6: "squamous_cell_carcinoma",
}

CLASS_NAMES = ["adenocarcinoma", "large_cell_carcinoma", "normal", "squamous_cell_carcinoma"]

print("Downloading Lung Cancer CT dataset from Hugging Face...")
ds = load_dataset("Mahadih534/Chest_CT-Scan_images-Dataset")

for split, hf_name in [("train", "train"), ("train", "validation"), ("test", "test")]:
    hf_split = ds[hf_name]
    print(f"\nProcessing {hf_name} -> local {split} ({len(hf_split)} samples)...")
    counts = {}
    for i, sample in enumerate(hf_split):
        label_idx = sample["label"]
        simple_label = LABEL_MAP[label_idx]
        if simple_label not in counts:
            counts[simple_label] = 0
        counts[simple_label] += 1
        target_dir = os.path.join(DATA_DIR, split, simple_label)
        os.makedirs(target_dir, exist_ok=True)
        img = sample["image"]
        if img.mode == "RGBA":
            img = img.convert("RGB")
        fname = f"{hf_name}_{i:04d}.jpg"
        fpath = os.path.join(target_dir, fname)
        if not os.path.exists(fpath):
            img.save(fpath, "JPEG")
    print(f"  Labels: {counts}")
    split_dir = os.path.join(DATA_DIR, split)
    total = 0
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(split_dir, cls)
        n = len([f for f in os.listdir(cls_dir) if f.endswith(".jpg")]) if os.path.exists(cls_dir) else 0
        print(f"  {split}/{cls}: {n} images")
        total += n
    print(f"  {split} total: {total}")

print("\nDataset ready!")
