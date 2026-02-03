import os
import json
import time
import random
import numpy as np
import torch
import comet_ml
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf

from vqvae_classifier import VQVAEClassifier


class NpyLabelDataset(Dataset):
    def __init__(self, x_path, labels_path, label_key="stats", label_map=None):
        self.x = np.load(x_path, allow_pickle=True)
        with np.load(labels_path) as d:
            labels_raw = d[label_key]

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


def build_dataloader(base_path, split, batchsize, label_key, label_map=None, revined_data=False, num_workers=1):
    if not revined_data:
        x_path = os.path.join(base_path, f"{split}_notrevin_x.npy")
        labels_path = os.path.join(base_path, f"{split}_notrevin_xlabels.npz")
    else:
        x_path = os.path.join(base_path, f"{split}_revin_x.npy")
        labels_path = os.path.join(base_path, f"{split}_revin_xlabels.npz")

    dataset = NpyLabelDataset(x_path, labels_path, label_key=label_key, label_map=label_map)
    loader = DataLoader(dataset, batch_size=batchsize, shuffle=(split == "train"), num_workers=num_workers)
    return loader, dataset.label_map


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

    # Build label map from train split
    train_loader, label_map = build_dataloader(
        config.base_path,
        "train",
        config.batchsize,
        config.label_key,
        label_map=None,
        revined_data=config.revined_data,
        num_workers=config.num_workers,
    )
    val_loader, _ = build_dataloader(
        config.val_path,
        "val",
        config.batchsize,
        config.label_key,
        label_map=label_map,
        revined_data=config.revined_data,
        num_workers=config.num_workers,
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

    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=config.classifier.learning_rate,
    )
    criterion = nn.CrossEntropyLoss()

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
    for epoch in range(config.classifier.num_epochs):
        model.train()
        train_losses = []
        train_accs = []

        for batch_x, batch_y in train_loader:
            batch_x = torch.tensor(batch_x, dtype=torch.float, device=device)
            batch_y = torch.tensor(batch_y, dtype=torch.long, device=device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_accs.append((logits.argmax(dim=1) == batch_y).float().mean().item())

        model.eval()
        val_losses = []
        val_accs = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = torch.tensor(batch_x, dtype=torch.float, device=device)
                batch_y = torch.tensor(batch_y, dtype=torch.long, device=device)

                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_losses.append(loss.item())
                val_accs.append((logits.argmax(dim=1) == batch_y).float().mean().item())

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={np.mean(train_losses):.4f} train_acc={np.mean(train_accs):.4f} | "
            f"val_loss={np.mean(val_losses):.4f} val_acc={np.mean(val_accs):.4f}"
        )
        
        if comet_logger:
            comet_logger.log_metric('train_loss_each_epoch', np.mean(train_losses), step=epoch)
            comet_logger.log_metric('train_acc_each_epoch', np.mean(train_accs), step=epoch)
            comet_logger.log_metric('val_loss_each_epoch', np.mean(val_losses), step=epoch)
            comet_logger.log_metric('val_acc_each_epoch', np.mean(val_accs), step=epoch)


    print("Total time:", round(time.time() - start_time, 2), "s")
    if comet_logger:
        comet_logger.end()

    if config.save_path:
        os.makedirs(config.save_path, exist_ok=True)
        torch.save(model, os.path.join(config.save_path, "vqvae_classifier.pth"))
        with open(os.path.join(config.save_path, "label_map.json"), "w") as f:
            json.dump(label_map, f, indent=2)


if __name__ == "__main__":
    main()
