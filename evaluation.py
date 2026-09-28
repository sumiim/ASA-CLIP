import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class GalleryDataset(Dataset):
    def __init__(self, photo_paths, labels, transform):
        self.photo_paths = photo_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.photo_paths)

    def __getitem__(self, idx):
        path = self.photo_paths[idx]
        label = self.labels[idx]
        instance_id = os.path.splitext(os.path.basename(path))[0]

        try:
            image = Image.open(path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception:
            image = torch.zeros(3, 224, 224)

        return image, label, instance_id


def _unpack_feature(result, index):
    if isinstance(result, (tuple, list)):
        return result[index]
    return result


@torch.no_grad()
def evaluate(model, test_loader, device, logger=None, use_amp=False, eval_metrics=None):
    model.eval()
    log_func = logger.info if logger else print
    eval_metrics = eval_metrics or ["map@all", "map@200", "p@100", "p@200"]
    test_dataset = test_loader.dataset

    gallery_photo_paths = []
    gallery_photo_labels = []
    for class_name, paths in test_dataset.photo_paths_by_class.items():
        if class_name not in test_dataset.class_to_idx:
            continue
        label = test_dataset.class_to_idx[class_name]
        gallery_photo_paths.extend(paths)
        gallery_photo_labels.extend([label] * len(paths))

    if hasattr(test_dataset, "transform_photo"):
        gallery_transform = test_dataset.transform_photo
    else:
        gallery_transform = test_dataset.transform

    gallery_dataset = GalleryDataset(gallery_photo_paths, gallery_photo_labels, gallery_transform)
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=min(test_loader.batch_size, 64),
        shuffle=False,
        num_workers=test_loader.num_workers,
        pin_memory=getattr(test_loader, "pin_memory", False),
        drop_last=False,
    )

    gallery_features = []
    gallery_ids = []
    for photos, _, ids in tqdm(gallery_loader, desc="Eval gallery", leave=False):
        photos = photos.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            if hasattr(model, "encode_photo"):
                feats = model.encode_photo(photos)
            else:
                dummy = torch.zeros_like(photos)
                feats = _unpack_feature(model(dummy, photos), 1)
        gallery_features.append(feats.detach().float().cpu())
        gallery_ids.extend(ids)

    if not gallery_features:
        return {}

    gallery_features = F.normalize(torch.cat(gallery_features, dim=0), p=2, dim=-1).to(device)
    gallery_labels = torch.tensor(gallery_photo_labels, dtype=torch.long, device=device)

    sketch_features = []
    sketch_labels = []
    sketch_ids = []
    for batch in tqdm(test_loader, desc="Eval sketches", leave=False):
        sketches = batch[0].to(device, non_blocking=True)
        labels = batch[2]
        ids = batch[3]
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            if hasattr(model, "encode_sketch"):
                feats = model.encode_sketch(sketches)
            else:
                dummy = torch.zeros_like(sketches)
                feats = _unpack_feature(model(sketches, dummy), 0)
        sketch_features.append(feats.detach().float().cpu())
        sketch_labels.append(labels)
        sketch_ids.extend(ids)

    sketch_features = F.normalize(torch.cat(sketch_features, dim=0), p=2, dim=-1).to(device)
    sketch_labels = torch.cat(sketch_labels, dim=0).long().to(device)

    unique_ids = sorted(set(gallery_ids + sketch_ids))
    id_map = {name: idx for idx, name in enumerate(unique_ids)}
    gallery_id_tensor = torch.tensor([id_map[x] for x in gallery_ids], dtype=torch.long, device=device)
    sketch_id_tensor = torch.tensor([id_map[x] for x in sketch_ids], dtype=torch.long, device=device)

    total_queries = sketch_features.size(0)
    eval_batch_size = 256
    max_k = 1
    if "p@100" in eval_metrics:
        max_k = max(max_k, 100)
    if "p@200" in eval_metrics or "map@200" in eval_metrics:
        max_k = max(max_k, 200)

    aps_all = []
    aps_200 = []
    precisions_100 = []
    precisions_200 = []

    ranks = torch.arange(1, gallery_features.size(0) + 1, device=device).float()
    topk_ranks = torch.arange(1, min(max_k, gallery_features.size(0)) + 1, device=device).float()

    for start in tqdm(range(0, total_queries, eval_batch_size), desc="Eval metrics", leave=False):
        end = start + eval_batch_size
        query_feats = sketch_features[start:end]
        query_labels = sketch_labels[start:end]

        sim = query_feats @ gallery_features.t()
        sorted_indices = torch.argsort(sim, dim=1, descending=True)
        ranked_labels = gallery_labels[sorted_indices]

        correct_class = ranked_labels.eq(query_labels.unsqueeze(1))
        total_relevant = gallery_labels.unsqueeze(0).eq(query_labels.unsqueeze(1)).sum(dim=1).clamp(min=1)

        if "p@100" in eval_metrics:
            k = min(100, correct_class.size(1))
            precisions_100.extend(correct_class[:, :k].float().mean(dim=1).cpu().numpy())
        if "p@200" in eval_metrics:
            k = min(200, correct_class.size(1))
            precisions_200.extend(correct_class[:, :k].float().mean(dim=1).cpu().numpy())

        if "map@all" in eval_metrics:
            precision = correct_class.float().cumsum(dim=1) / ranks[: correct_class.size(1)]
            ap = (precision * correct_class.float()).sum(dim=1) / total_relevant
            aps_all.extend(ap.cpu().numpy())

        if "map@200" in eval_metrics:
            k = min(200, correct_class.size(1))
            correct_k = correct_class[:, :k].float()
            precision_k = correct_k.cumsum(dim=1) / topk_ranks[:k]
            num_relevant_k = correct_k.sum(dim=1).clamp(min=1)
            ap_k = (precision_k * correct_k).sum(dim=1) / num_relevant_k
            aps_200.extend(ap_k.cpu().numpy())

    results = {}
    if "map@all" in eval_metrics:
        results["map@all"] = 100.0 * float(np.mean(aps_all))
    if "map@200" in eval_metrics:
        results["map@200"] = 100.0 * float(np.mean(aps_200))
    if "p@100" in eval_metrics:
        results["p@100"] = 100.0 * float(np.mean(precisions_100))
    if "p@200" in eval_metrics:
        results["p@200"] = 100.0 * float(np.mean(precisions_200))

    log_func("Evaluation uses pure visual features only.")
    model.train()
    return results
