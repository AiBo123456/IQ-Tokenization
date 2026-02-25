import os
import json
import time
import random
import numpy as np
import torch
import comet_ml
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from vqvae_classifier import VQVAEClassifier


# class NpyLabelDataset(Dataset):
#     def __init__(self, x_path, labels_path, label_key="stats", label_map=None):
#         self.x = np.load(x_path, allow_pickle=True)
#         with np.load(labels_path) as d:
#             labels_raw = d[label_key]

#         if label_map is None:
#             unique = sorted(list(set(labels_raw.tolist())))
#             label_map = {v: i for i, v in enumerate(unique)}

#         self.label_map = label_map
#         self.y = np.array([label_map[v] for v in labels_raw], dtype=np.int64)

#         assert len(self.x) == len(self.y), "x and labels length mismatch"

#     def __len__(self):
#         return len(self.x)

#     def __getitem__(self, idx):
#         return self.x[idx], self.y[idx]


class NpyLabelDataset(Dataset):
    def __init__(self, x_path, labels_path, label_key="stats", label_map=None):
        self.x = np.load(x_path, allow_pickle=True)
        labels_raw = np.load(labels_path, allow_pickle=True)

        if label_map is None:
            unique = sorted(list(set(labels_raw.tolist())))
            label_map = {v: i for i, v in enumerate(unique)}

        self.label_map = label_map
        self.y = np.array([label_map[v] for v in labels_raw], dtype=np.int64)

        assert len(self.x) == len(self.y), "x and labels length mismatch"

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class ArrayLabelDataset(Dataset):
    def __init__(
        self,
        x,
        y,
        augment=False,
        noise_std=0.0,
        time_mask_ratio=0.0,
        channel_dropout=0.0,
    ):
        self.x = np.asarray(x, dtype=np.float32)
        self.y = y
        self.augment = bool(augment)
        self.noise_std = float(noise_std)
        self.time_mask_ratio = float(time_mask_ratio)
        self.channel_dropout = float(channel_dropout)

        assert len(self.x) == len(self.y), "x and labels length mismatch"

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.augment:
            if self.noise_std > 0:
                x = x + np.random.normal(0.0, self.noise_std, size=x.shape).astype(np.float32)
            if self.channel_dropout > 0 and x.ndim >= 2:
                keep = (np.random.rand(x.shape[0]) >= self.channel_dropout).astype(np.float32)
                if np.any(keep == 0):
                    x = x.copy()
                    x *= keep[:, None]
            if self.time_mask_ratio > 0 and x.ndim >= 2:
                t = x.shape[-1]
                mask_len = int(round(t * self.time_mask_ratio))
                if mask_len > 0:
                    start = np.random.randint(0, max(1, t - mask_len + 1))
                    x = x.copy()
                    x[..., start : start + mask_len] = 0.0
        return x, self.y[idx]


def load_split_arrays(base_path, split, revined_data=False, prn_id=23):
    if not revined_data:
        x_path = os.path.join(base_path, f"{split}_notrevin_x.npy")
    else:
        x_path = os.path.join(base_path, f"{split}_revin_x.npy")
    labels_path = os.path.join(base_path, f"{split}_x_labels.npy")

    if not (os.path.exists(x_path) and os.path.exists(labels_path)):
        return None

    x = np.load(x_path, allow_pickle=True)
    labels_raw = np.load(labels_path, allow_pickle=True)
    return x, labels_raw[:, prn_id-1]


def build_combined_dataloaders(
    base_path,
    batchsize,
    revined_data=False,
    prn_id=23,
    num_workers=1,
    seed=0,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    balanced_sampling=True,
    train_noise_std=0.0,
    train_time_mask_ratio=0.0,
    train_channel_dropout=0.0,
):
    split_data = []
    for split in ("train", "val", "test"):
        data = load_split_arrays(base_path, split, revined_data=revined_data, prn_id=prn_id)
        if data is not None:
            split_data.append(data)

    if not split_data:
        raise FileNotFoundError("No train/val/test split files found under base_path.")

    x_all = np.concatenate([d[0] for d in split_data], axis=0)
    labels_raw_all = np.concatenate([d[1] for d in split_data], axis=0)
    values, counts = np.unique(labels_raw_all, return_counts=True)

    # # Get unique elements and their counts
    # unique_elements, counts = np.unique(labels_raw_all, return_counts=True)

    # # Print results
    # for elem, count in zip(unique_elements, counts):
    #     print(f"Value: {elem}, Frequency: {count}")
    # 0/0

    # freq = dict(zip(values, counts))

    unique = sorted(list(set(labels_raw_all.tolist())))
    label_map = {v: i for i, v in enumerate(unique)}
    y_all = np.array([label_map[v] for v in labels_raw_all], dtype=np.int64)

    total = len(y_all)
    if train_ratio + val_ratio + test_ratio <= 0:
        raise ValueError("Split ratios must sum to a positive value.")

    ratio_sum = train_ratio + val_ratio + test_ratio
    ratios = np.array([train_ratio, val_ratio, test_ratio], dtype=np.float64) / ratio_sum
    test_size = float(ratios[2])
    val_size_within_trainval = float(ratios[1] / (ratios[0] + ratios[1])) if (ratios[0] + ratios[1]) > 0 else 0.0

    if test_size > 0:
        x_trainval, x_test, y_trainval, y_test = train_test_split(
            x_all,
            y_all,
            test_size=test_size,
            random_state=seed,
            stratify=y_all,
        )
    else:
        x_trainval, y_trainval = x_all, y_all
        x_test = np.zeros((0, *x_all.shape[1:]), dtype=x_all.dtype)
        y_test = np.zeros((0,), dtype=y_all.dtype)

    if val_size_within_trainval > 0:
        x_train, x_val, y_train, y_val = train_test_split(
            x_trainval,
            y_trainval,
            test_size=val_size_within_trainval,
            random_state=seed,
            stratify=y_trainval,
        )
    else:
        x_train, y_train = x_trainval, y_trainval
        x_val = np.zeros((0, *x_all.shape[1:]), dtype=x_all.dtype)
        y_val = np.zeros((0,), dtype=y_all.dtype)

    train_ds = ArrayLabelDataset(
        x_train,
        y_train,
        augment=True,
        noise_std=train_noise_std,
        time_mask_ratio=train_time_mask_ratio,
        channel_dropout=train_channel_dropout,
    )
    val_ds = ArrayLabelDataset(x_val, y_val, augment=False)
    test_ds = ArrayLabelDataset(x_test, y_test, augment=False) if len(x_test) > 0 else None

    generator = torch.Generator().manual_seed(seed)
    if balanced_sampling and len(y_train) > 0:
        class_counts = np.bincount(y_train)
        sample_weights = (1.0 / np.maximum(class_counts, 1))[y_train]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
        train_loader = DataLoader(train_ds, batch_size=batchsize, sampler=sampler, num_workers=num_workers)
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batchsize,
            shuffle=True,
            num_workers=num_workers,
            generator=generator,
        )

    val_loader = DataLoader(val_ds, batch_size=batchsize, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batchsize, shuffle=False, num_workers=num_workers) if test_ds is not None else None

    return train_loader, val_loader, test_loader, label_map, y_train


def run_eval(model, loader, device, criterion):
    model.eval()
    losses = []
    probs = []
    labels = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device=device, dtype=torch.float32).detach()
            batch_y = batch_y.to(device=device, dtype=torch.long).detach()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            losses.append(loss.item())
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.append(batch_y.cpu().numpy())

    probs = np.concatenate(probs, axis=0)
    labels = np.concatenate(labels, axis=0)
    acc = float((probs.argmax(axis=1) == labels).mean())
    try:
        if probs.shape[1] == 2:
            auc = roc_auc_score(labels, probs[:, 1])
        else:
            auc = roc_auc_score(labels, probs, multi_class="ovr")
    except ValueError:
        auc = float("nan")
    return float(np.mean(losses)), acc, float(auc)


def reinit_classifier_parameters(model, include_vqvae=False):
    vqvae_modules = set(model.vqvae.modules())
    for module in model.modules():
        if (not include_vqvae) and (module in vqvae_modules):
            continue
        reset_fn = getattr(module, "reset_parameters", None)
        if callable(reset_fn):
            reset_fn()
    if getattr(model, "cls_token", None) is not None:
        nn.init.normal_(model.cls_token, mean=0.0, std=0.02)


def main():
    config = OmegaConf.load("vqvae_classifier_config.yaml")

    seed = config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available() and config.model_init_num_gpus >= 0:
        device = f"cuda:{config.model_init_num_gpus}"
    else:
        device = "cpu"

    # Load pretrained VQ-VAE
    vqvae = torch.load(config.classifier.pretrained_vqvae_path, map_location=device, weights_only=False)

    # Load all splits together, then randomly resample train/val/test
    train_ratio = config.get("train_ratio", 0.8)
    val_ratio = config.get("val_ratio", 0.1)
    test_ratio = config.get("test_ratio", 0.1)

    train_loader, val_loader, test_loader, label_map, y_train = build_combined_dataloaders(
        config.base_path,
        config.batchsize,
        revined_data=config.revined_data,
        prn_id=int(config.get("prn_id", 23)),
        num_workers=config.num_workers,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        balanced_sampling=bool(config.classifier.get("balanced_sampling", True)),
        train_noise_std=float(config.classifier.get("train_noise_std", 0.0)),
        train_time_mask_ratio=float(config.classifier.get("train_time_mask_ratio", 0.0)),
        train_channel_dropout=float(config.classifier.get("train_channel_dropout", 0.0)),
    )

    num_classes = config.classifier.num_classes or len(label_map)

    model = VQVAEClassifier(
        vqvae=vqvae,
        num_classes=num_classes,
        mlp_hidden=config.classifier.mlp_hidden,
        dropout=config.classifier.dropout,
        pooling=config.classifier.pooling,
        arch=config.classifier.arch,
        d_model=config.classifier.d_model,
        nhead=config.classifier.nhead,
        num_layers=config.classifier.num_layers,
        dim_feedforward=config.classifier.dim_feedforward,
        reinit_model_params=config.reinit_model_params,
    ).to(device)

    if config.reinit_model_params:
        reinit_classifier_parameters(model, include_vqvae=True)

    learning_rate = float(config.classifier.learning_rate)
    weight_decay = float(config.classifier.get("weight_decay", 1e-3))
    label_smoothing = float(config.classifier.get("label_smoothing", 0.0))
    class_weighting = bool(config.classifier.get("class_weighting", False))
    scheduler_patience = int(config.classifier.get("scheduler_patience", 3))
    scheduler_factor = float(config.classifier.get("scheduler_factor", 0.5))
    min_lr = float(config.classifier.get("min_lr", 1e-6))
    early_stop_patience = int(config.classifier.get("early_stop_patience", 10))
    max_grad_norm = float(config.classifier.get("max_grad_norm", 1.0))

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=min_lr,
    )
    if class_weighting:
        class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
        class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
        class_weights = class_weights / class_weights.mean()
        class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # set up comet logger
    if config.comet_log:
        # Create an experiment with your api key
        comet_logger = comet_ml.Experiment(
            api_key=config['comet_config']['api_key'],
            project_name=config['comet_config']['project_name'],
            workspace=config['comet_config']['workspace'],
        )
        comet_logger.add_tag(config.comet_tag)
        comet_logger.set_name(config.comet_name)
    else:
        print('PROBLEM: not saving to comet')
        comet_logger = None

    start_time = time.time()
    best_state = None
    best_epoch = -1
    best_val_loss = float("inf")
    best_val_auc = float("-inf")
    stale_epochs = 0
    for epoch in range(config.classifier.num_epochs):
        model.train()
        train_losses = []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device=device, dtype=torch.float32).detach()
            batch_y = batch_y.to(device=device, dtype=torch.long).detach()

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            train_losses.append(loss.item())

        train_loss, train_acc, train_auc = run_eval(model, train_loader, device, criterion)
        val_loss, val_acc, val_auc = run_eval(model, val_loader, device, criterion)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_auc={train_auc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_auc={val_auc:.4f}"
        )

        improved = False
        if not np.isnan(val_auc) and val_auc > best_val_auc + 1e-6:
            improved = True
        elif np.isnan(val_auc) and val_loss < best_val_loss - 1e-6:
            improved = True

        if improved:
            best_val_loss = val_loss
            if not np.isnan(val_auc):
                best_val_auc = val_auc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= early_stop_patience:
                print(f"Early stopping at epoch {epoch:03d}. Best epoch={best_epoch:03d} val_loss={best_val_loss:.4f}")
                break
        
        if comet_logger:
            comet_logger.log_metric('train_loss_each_epoch', train_loss, step=epoch)
            comet_logger.log_metric('train_acc_each_epoch', train_acc, step=epoch)
            comet_logger.log_metric('train_auc_each_epoch', train_auc, step=epoch)
            comet_logger.log_metric('val_loss_each_epoch', val_loss, step=epoch)
            comet_logger.log_metric('val_acc_each_epoch', val_acc, step=epoch)
            comet_logger.log_metric('val_auc_each_epoch', val_auc, step=epoch)
            comet_logger.log_metric('learning_rate', optimizer.param_groups[0]["lr"], step=epoch)


    print("Total time:", round(time.time() - start_time, 2), "s")
    if comet_logger:
        comet_logger.end()

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Loaded best checkpoint from epoch {best_epoch:03d} (val_loss={best_val_loss:.4f}, val_auc={best_val_auc:.4f})")

    if test_loader is not None:
        test_loss, test_acc, test_auc = run_eval(model, test_loader, device, criterion)
        print(f"Final test | loss={test_loss:.4f} acc={test_acc:.4f} auc={test_auc:.4f}")

    if config.save_path:
        os.makedirs(config.save_path, exist_ok=True)
        torch.save(model, os.path.join(config.save_path, "vqvae_classifier.pth"))
        with open(os.path.join(config.save_path, "label_map.json"), "w") as f:
            json.dump(label_map, f, indent=2)


if __name__ == "__main__":
    main()
