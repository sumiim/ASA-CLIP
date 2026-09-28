import os
import random
import re
from glob import glob

import torch
from PIL import Image
from torch.utils.data import Dataset


DATASET_SPLIT_SEED = 42

SKETCHY_V2_TEST_CLASSES = [
    "bat",
    "cabin",
    "cow",
    "dolphin",
    "door",
    "giraffe",
    "helicopter",
    "mouse",
    "pear",
    "raccoon",
    "rhinoceros",
    "saw",
    "scissors",
    "seagull",
    "skyscraper",
    "songbird",
    "sword",
    "tree",
    "wheelchair",
    "windmill",
    "window",
]

QUICKDRAW_CLASSES = {
    "train": [
        "airplane", "alarm_clock", "ant", "apple", "axe", "bandage", "bear", "bee",
        "bench", "bicycle", "bus", "butterfly", "camel", "candle", "car", "castle",
        "cat", "chair", "chandelier", "church", "couch", "crab", "crocodilian",
        "cruise_ship", "cup", "dog", "duck", "eiffel_tower", "elephant", "eyeglasses",
        "fish", "flower", "guitar", "hammer", "hat", "hedgehog", "horse",
        "hot-air_balloon", "hurricane", "kangaroo", "knife", "lighthouse", "lion",
        "lobster", "map", "monkey", "moon", "motorcycle", "mountain", "mushroom",
        "owl", "parrot", "penguin", "pickup_truck", "pig", "pineapple", "pizza",
        "rabbit", "rifle", "sailboat", "saxophone", "scorpion", "sea_turtle",
        "sheep", "shoe", "sink", "skateboard", "snail", "snake", "soccer_ball",
        "spider", "spoon", "squirrel", "swan", "teapot", "teddy_bear", "train",
        "umbrella", "violin", "wine_bottle",
    ],
    "test": [
        "bat", "cow", "dolphin", "door", "giraffe", "helicopter", "mouse",
        "raccoon", "rhinoceros", "saw", "scissors", "skyscraper", "tree",
        "windmill", "feather", "campfire", "palm_tree", "fire_hydrant", "bread",
        "beach", "megaphone", "cactus", "zebra", "tiger", "shark", "frog",
        "banana", "cake", "hamburger", "fan",
    ],
}

# Held-out seen classes used only for model selection. Official unseen test is unchanged.
VAL_CLASS_COUNTS = {
    "sketchy": 10,
    "tuberlin": 20,
    "quickdraw": 8,
}


def holdout_val_classes(seen_classes, dataset_name, seed=DATASET_SPLIT_SEED):
    """Split a fixed val set from already-determined seen classes.

    The draw is independent of --seed so train/val/test Dataset instances match.
    """
    if dataset_name not in VAL_CLASS_COUNTS:
        raise ValueError(f"No val-class count for dataset: {dataset_name}")
    n_val = VAL_CLASS_COUNTS[dataset_name]
    ordered = sorted(seen_classes)
    if n_val >= len(ordered):
        raise ValueError(
            f"Cannot hold out {n_val} val classes from {len(ordered)} seen classes ({dataset_name})."
        )
    val = sorted(random.Random(seed).sample(ordered, n_val))
    train = sorted(set(ordered) - set(val))
    return train, val


class BaseSketchPhotoDataset(Dataset):
    def __init__(self, transform=None, transform_sketch=None, transform_photo=None):
        if transform_sketch is not None and transform_photo is not None:
            self.transform_sketch = transform_sketch
            self.transform_photo = transform_photo
            self.transform = transform_photo
        elif transform is not None:
            self.transform_sketch = transform
            self.transform_photo = transform
            self.transform = transform
        else:
            raise ValueError("A transform or transform_sketch/transform_photo pair is required.")

    @staticmethod
    def normalize_name(name):
        name = name.lower()
        name = re.sub(r"\s*\((.*?)\)\s*", r"_\1", name)
        name = name.replace("-", "_").replace(" ", "_")
        name = re.sub(r"_+", "_", name)
        return name

    @staticmethod
    def image_files(root):
        files = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.JPEG"):
            files.extend(glob(os.path.join(root, ext)))
        return files

    def _load_image_pair(self, sketch_path, photo_path):
        try:
            sketch_img = Image.open(sketch_path).convert("RGB")
            sketch = self.transform_sketch(sketch_img)
        except Exception:
            sketch = torch.zeros(3, 224, 224)

        try:
            photo_img = Image.open(photo_path).convert("RGB")
            photo = self.transform_photo(photo_img)
        except Exception:
            photo = torch.zeros(3, 224, 224)

        return sketch, photo


class QuickDrawDataset(BaseSketchPhotoDataset):
    """QuickDraw-Ext and TU-Berlin-Ext category-level datasets."""

    def __init__(
        self,
        root_dir,
        mode="train",
        dataset_name="quickdraw",
        transform=None,
        transform_sketch=None,
        transform_photo=None,
        logger=None,
        gzs=False,
        gzs_seen_ratio=0.2,
        **kwargs,
    ):
        super().__init__(transform, transform_sketch, transform_photo)
        if mode not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported mode: {mode}. Expected train/val/test.")
        self.root_dir = root_dir
        self.mode = mode
        self.dataset_name = dataset_name
        self.gzs = gzs
        self.gzs_seen_ratio = gzs_seen_ratio
        self.log_func = logger.info if logger else print
        self.seen_classes_pool = []
        self.val_classes_pool = []

        if dataset_name == "quickdraw":
            self.sketch_dir = os.path.join(root_dir, "sketches")
            self.photo_dir = os.path.join(root_dir, "photos")
        elif dataset_name == "tuberlin":
            self.sketch_dir = os.path.join(root_dir, "png_ready")
            self.photo_dir = os.path.join(root_dir, "ImageResized_ready")
        else:
            raise ValueError(f"Unsupported dataset_name for QuickDrawDataset: {dataset_name}")

        self.train_classes_pool = []
        self.test_classes_pool = []
        self.class_to_idx = {}
        self.sketch_paths = []
        self.photo_paths_by_class = {}
        self._load_data()

    def _split_classes(self, valid_pair_classes, photo_counts):
        if self.dataset_name == "quickdraw":
            train = [
                self.normalize_name(c)
                for c in QUICKDRAW_CLASSES["train"]
                if self.normalize_name(c) in valid_pair_classes
            ]
            test = [
                self.normalize_name(c)
                for c in QUICKDRAW_CLASSES["test"]
                if self.normalize_name(c) in valid_pair_classes
            ]
            return sorted(train), sorted(test)

        test_candidates, train_candidates = [], []
        for cls in sorted(valid_pair_classes):
            if photo_counts.get(cls, 0) >= 400:
                test_candidates.append(cls)
            else:
                train_candidates.append(cls)
        random.Random(DATASET_SPLIT_SEED).shuffle(test_candidates)
        test = sorted(test_candidates[: min(30, len(test_candidates))])
        train = sorted(set(test_candidates[len(test):]).union(train_candidates))
        return train, test

    def _load_data(self):
        self.log_func(f"Loading {self.dataset_name} ({self.mode})")
        sketch_dirs = {
            self.normalize_name(os.path.basename(path)): path
            for path in glob(os.path.join(self.sketch_dir, "*"))
            if os.path.isdir(path)
        }
        photo_dirs = {
            self.normalize_name(os.path.basename(path)): path
            for path in glob(os.path.join(self.photo_dir, "*"))
            if os.path.isdir(path)
        }
        valid_classes = set(sketch_dirs).intersection(photo_dirs)
        photo_counts = {cls: len(self.image_files(photo_dirs[cls])) for cls in valid_classes}

        seen_classes, self.test_classes_pool = self._split_classes(valid_classes, photo_counts)
        self.seen_classes_pool = list(seen_classes)
        self.train_classes_pool, self.val_classes_pool = holdout_val_classes(
            self.seen_classes_pool, self.dataset_name
        )
        all_classes = sorted(set(self.seen_classes_pool).union(self.test_classes_pool))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}
        self.log_func(
            f"Split classes: train={len(self.train_classes_pool)}, "
            f"val={len(self.val_classes_pool)}, test={len(self.test_classes_pool)}"
        )

        query_classes, gallery_classes = self._classes_for_mode()

        for cls in query_classes:
            self.sketch_paths.extend((path, cls) for path in self.image_files(sketch_dirs[cls]))

        for cls in gallery_classes:
            files = self.image_files(photo_dirs[cls])
            if files:
                self.photo_paths_by_class[cls] = files

        self.log_func(
            f"Loaded queries={len(self.sketch_paths)}, gallery_classes={len(self.photo_paths_by_class)}"
        )

    def _classes_for_mode(self):
        if self.mode == "train":
            return self.train_classes_pool, self.train_classes_pool
        if self.mode == "val":
            return self.val_classes_pool, self.val_classes_pool
        query_classes = self.test_classes_pool
        if self.gzs:
            num_seen = int(len(self.seen_classes_pool) * self.gzs_seen_ratio)
            seen = random.Random(DATASET_SPLIT_SEED).sample(self.seen_classes_pool, num_seen)
            gallery_classes = sorted(set(query_classes).union(seen))
        else:
            gallery_classes = query_classes
        return query_classes, gallery_classes

    def __len__(self):
        return len(self.sketch_paths)

    def __getitem__(self, idx):
        sketch_path, class_name = self.sketch_paths[idx]
        if class_name not in self.photo_paths_by_class:
            raise RuntimeError(f"No gallery photos found for class: {class_name}")
        photo_path = random.choice(self.photo_paths_by_class[class_name])
        sketch, photo = self._load_image_pair(sketch_path, photo_path)
        label = torch.tensor(self.class_to_idx[class_name], dtype=torch.long)
        instance_id = os.path.splitext(os.path.basename(sketch_path))[0]
        return sketch, photo, label, instance_id


class SketchyDataset(BaseSketchPhotoDataset):
    """Sketchy-Ext category-level dataset loader."""

    def __init__(
        self,
        root_dir,
        mode="train",
        split_mode="sketchy2",
        task_type="category",
        transform=None,
        transform_sketch=None,
        transform_photo=None,
        logger=None,
        gzs=False,
        gzs_seen_ratio=0.2,
        **kwargs,
    ):
        if mode not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported mode: {mode}. Expected train/val/test.")
        super().__init__(transform, transform_sketch, transform_photo)
        self.root_dir = root_dir
        self.mode = mode
        self.dataset_name = "sketchy"
        self.split_mode = split_mode
        self.task_type = "category"
        self.gzs = gzs
        self.gzs_seen_ratio = gzs_seen_ratio
        self.log_func = logger.info if logger else print

        self.sketch_root = os.path.join(root_dir, "sketch", "tx_000000000000")
        self.paired_photo_root = os.path.join(root_dir, "photo", "tx_000000000000")
        self.extended_photo_root = os.path.join(root_dir, "extended_photo")
        if not os.path.exists(self.sketch_root):
            self.sketch_root = os.path.join(root_dir, "sketch")

        all_classes = sorted(
            d for d in os.listdir(self.sketch_root) if os.path.isdir(os.path.join(self.sketch_root, d))
        )
        seen_classes, test_classes = self._get_splits(all_classes)
        self.seen_classes_pool = list(seen_classes)
        self.test_classes_pool = list(test_classes)
        self.train_classes_pool, self.val_classes_pool = holdout_val_classes(
            self.seen_classes_pool, "sketchy"
        )
        all_split_classes = sorted(set(self.seen_classes_pool).union(self.test_classes_pool))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(all_split_classes)}
        self.sketch_paths = []
        self.photo_paths_by_class = {}
        self._load_data()

    def _get_splits(self, all_classes):
        normalized = {
            cls.lower().replace("_", "").replace("(", "").replace(")", ""): cls
            for cls in all_classes
        }
        if self.split_mode == "sketchy2":
            test_names = []
            for target in SKETCHY_V2_TEST_CLASSES:
                key = target.lower().replace("_", "")
                match = normalized.get(key)
                if match:
                    test_names.append(match)
            test_set = set(test_names)
            train_set = set(all_classes) - test_set
            return sorted(train_set), sorted(test_set)

        shuffled = sorted(all_classes)
        random.Random(DATASET_SPLIT_SEED).shuffle(shuffled)
        return sorted(shuffled[25:]), sorted(shuffled[:25])

    def _load_data(self):
        self.log_func(f"Loading Sketchy ({self.mode})")
        self.log_func(
            f"Split classes: train={len(self.train_classes_pool)}, "
            f"val={len(self.val_classes_pool)}, test={len(self.test_classes_pool)}"
        )
        if self.mode == "train":
            query_classes = self.train_classes_pool
            gallery_classes = self.train_classes_pool
        elif self.mode == "val":
            query_classes = self.val_classes_pool
            gallery_classes = self.val_classes_pool
        elif self.mode == "test":
            query_classes = self.test_classes_pool
            if self.gzs:
                num_seen = int(len(self.seen_classes_pool) * self.gzs_seen_ratio)
                seen = random.Random(DATASET_SPLIT_SEED).sample(self.seen_classes_pool, num_seen)
                gallery_classes = sorted(set(self.test_classes_pool).union(seen))
            else:
                gallery_classes = self.test_classes_pool
        else:
            raise ValueError(f"Unsupported mode: {self.mode}. Expected train/val/test.")

        for cls in sorted(set(query_classes).union(gallery_classes)):
            sketch_dir = os.path.join(self.sketch_root, cls)
            paired_dir = os.path.join(self.paired_photo_root, cls)
            extended_dir = os.path.join(self.extended_photo_root, cls)

            if cls in query_classes:
                sketches = self.image_files(sketch_dir)
                self.sketch_paths.extend((path, cls) for path in sketches)

            if cls in gallery_classes:
                photos = []
                if os.path.exists(extended_dir):
                    photos.extend(self.image_files(extended_dir))
                if os.path.exists(paired_dir):
                    photos.extend(self.image_files(paired_dir))
                if photos:
                    self.photo_paths_by_class[cls] = photos

        self.log_func(f"Loaded queries={len(self.sketch_paths)}, gallery_classes={len(self.photo_paths_by_class)}")

    def __len__(self):
        return len(self.sketch_paths)

    def __getitem__(self, idx):
        sketch_path, class_name = self.sketch_paths[idx]
        if class_name not in self.photo_paths_by_class:
            raise RuntimeError(f"No gallery photos found for class: {class_name}")
        photo_path = random.choice(self.photo_paths_by_class[class_name])

        sketch, photo = self._load_image_pair(sketch_path, photo_path)
        label = torch.tensor(self.class_to_idx[class_name], dtype=torch.long)
        instance_id = os.path.splitext(os.path.basename(sketch_path))[0]
        return sketch, photo, label, instance_id
