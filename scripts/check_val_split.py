#!/usr/bin/env python3
"""CPU-only check that train/val/test class splits are disjoint and stable."""
from __future__ import annotations

import os
import sys

from torchvision import transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dataset import VAL_CLASS_COUNTS, QuickDrawDataset, SketchyDataset  # noqa: E402


IDENTITY = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

DATA_DIR_MAP = {
    "sketchy": "Sketchy_extended",
    "tuberlin": "TUBerlin",
    "quickdraw": "quickdraw_extended",
}


class _SilentLogger:
    def info(self, *args, **kwargs):
        pass


def build(dataset_name, root, mode):
    cls = SketchyDataset if dataset_name == "sketchy" else QuickDrawDataset
    return cls(
        root_dir=root,
        mode=mode,
        dataset_name=dataset_name,
        split_mode="sketchy2",
        transform=IDENTITY,
        logger=_SilentLogger(),
        gzs=False,
    )


def check_dataset(dataset_name, data_path):
    root = os.path.join(data_path, DATA_DIR_MAP[dataset_name])
    if not os.path.isdir(root):
        print(f"[SKIP] {dataset_name}: missing {root}")
        return False
    train = build(dataset_name, root, "train")
    val = build(dataset_name, root, "val")
    test = build(dataset_name, root, "test")

    assert train.val_classes_pool == val.val_classes_pool == test.val_classes_pool
    assert train.train_classes_pool == val.train_classes_pool == test.train_classes_pool
    assert train.test_classes_pool == val.test_classes_pool == test.test_classes_pool
    assert train.class_to_idx == val.class_to_idx == test.class_to_idx

    train_set, val_set, test_set = (
        set(train.train_classes_pool),
        set(train.val_classes_pool),
        set(train.test_classes_pool),
    )
    assert not train_set & val_set, train_set & val_set
    assert not val_set & test_set, val_set & test_set
    assert not train_set & test_set, train_set & test_set
    expected = VAL_CLASS_COUNTS[dataset_name]
    assert len(val_set) == expected, (dataset_name, len(val_set), expected)
    assert len(val.sketch_paths) > 0
    assert len(val.photo_paths_by_class) == expected

    print(
        f"[OK] {dataset_name}: train={len(train_set)} val={len(val_set)} "
        f"test={len(test_set)} class_to_idx={len(train.class_to_idx)} "
        f"val_queries={len(val.sketch_paths)}"
    )
    print("     val classes:", ", ".join(train.val_classes_pool))
    return True


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "..", "data")
    data_path = os.path.abspath(data_path)
    print(f"data_path={data_path}")
    ok = True
    for name in ("sketchy", "tuberlin", "quickdraw"):
        try:
            ok = check_dataset(name, data_path) and ok
        except Exception as exc:
            ok = False
            print(f"[FAIL] {name}: {exc}")
            raise
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
