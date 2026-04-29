import argparse
import copy
import json
import os
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from classify_clean_vs_ofdm_svm import (  # noqa: E402
    average_numeric_structure,
    build_labeled_split,
    combine_splits,
    load_model,
    print_stage,
    resolve_feature_inputs,
    set_seed,
    summarize_feature_dim,
    transform_features_for_classification,
)


class BinaryMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer size.")

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    # nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ConvSequenceClassifier(nn.Module):
    def __init__(
        self,
        input_channels: int,
        conv_channels: list[int],
        head_hidden_dims: list[int],
        dropout: float,
        pooled_steps: int = 4,
    ) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError(f"input_channels must be positive, got {input_channels}")
        if not conv_channels:
            raise ValueError("conv_channels must contain at least one channel size.")

        blocks: list[nn.Module] = []
        prev_channels = input_channels
        for layer_idx, out_channels in enumerate(conv_channels):
            kernel_size = 9 if layer_idx == 0 else 5
            stride = 1 if layer_idx == 0 else 2
            blocks.extend(
                [
                    nn.Conv1d(
                        prev_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=kernel_size // 2,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.GELU(),
                    # nn.Dropout1d(dropout),
                ]
            )
            prev_channels = out_channels

        self.backbone = nn.Sequential(*blocks)
        self.avg_pool = nn.AdaptiveAvgPool1d(pooled_steps)
        self.max_pool = nn.AdaptiveMaxPool1d(pooled_steps)

        head_layers: list[nn.Module] = []
        head_input_dim = prev_channels * pooled_steps * 2
        prev_dim = head_input_dim
        for hidden_dim in head_hidden_dims:
            head_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    # nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        head_layers.append(nn.Linear(prev_dim, 1))
        self.head = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        pooled = torch.cat([self.avg_pool(features), self.max_pool(features)], dim=1)
        return self.head(torch.flatten(pooled, start_dim=1)).squeeze(-1)


def parse_hidden_dims(hidden_dims: str) -> list[int]:
    dims = [int(part.strip()) for part in hidden_dims.split(",") if part.strip()]
    if not dims:
        raise ValueError("Expected at least one hidden dimension, e.g. '256,128'.")
    if any(dim <= 0 for dim in dims):
        raise ValueError(f"All hidden dimensions must be positive, got: {dims}")
    return dims


def prepare_features_for_model(
    clean_features: np.ndarray,
    anomaly_features: np.ndarray,
    feature_set: str,
    code_id_encoding: str,
) -> tuple[np.ndarray, np.ndarray, list[int], int, str]:
    if feature_set in {"raw", "codes", "unquantized_codes"}:
        transformed_shape = list(clean_features.shape[1:])
        transformed_dim = summarize_feature_dim(clean_features)
        return (
            clean_features.astype(np.float32),
            anomaly_features.astype(np.float32),
            transformed_shape,
            transformed_dim,
            "structured_sequence",
        )

    return transform_features_for_classification(
        clean_features=clean_features,
        anomaly_features=anomaly_features,
        feature_set=feature_set,
        code_id_encoding=code_id_encoding,
    )


def make_loader(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(features.astype(np.float32)),
        torch.from_numpy(labels.astype(np.float32)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    threshold: float,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_logits: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            probs = torch.sigmoid(logits)
            batch_size = int(batch_x.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size

            all_logits.append(logits.detach().cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())
            all_labels.append(batch_y.detach().cpu().numpy())

    labels = np.concatenate(all_labels, axis=0).astype(np.int64)
    probs = np.concatenate(all_probs, axis=0).astype(np.float32)
    logits = np.concatenate(all_logits, axis=0).astype(np.float32)
    preds = (probs >= threshold).astype(np.int64)

    metrics: dict[str, Any] = {
        "loss": float(total_loss / max(total_count, 1)),
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        "classification_report": classification_report(
            labels,
            preds,
            target_names=["clean", "anomaly"],
            output_dict=True,
            zero_division=0,
        ),
        "score_mean_clean": float(np.mean(probs[labels == 0])) if np.any(labels == 0) else 0.0,
        "score_mean_anomaly": float(np.mean(probs[labels == 1])) if np.any(labels == 1) else 0.0,
        "logit_mean_clean": float(np.mean(logits[labels == 0])) if np.any(labels == 0) else 0.0,
        "logit_mean_anomaly": float(np.mean(logits[labels == 1])) if np.any(labels == 1) else 0.0,
    }
    if np.unique(labels).size > 1:
        metrics["roc_auc"] = float(roc_auc_score(labels, probs))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def collect_probabilities(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            probs = torch.sigmoid(model(batch_x))
            all_probs.append(probs.detach().cpu().numpy())
            all_labels.append(batch_y.detach().cpu().numpy())

    return (
        np.concatenate(all_probs, axis=0).astype(np.float32),
        np.concatenate(all_labels, axis=0).astype(np.int64),
    )


def find_best_threshold(probs: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    candidate_thresholds = np.linspace(0.05, 0.95, 37, dtype=np.float32)
    best_threshold = 0.5
    best_score = float("-inf")

    for threshold in candidate_thresholds:
        preds = (probs >= threshold).astype(np.int64)
        score = float(balanced_accuracy_score(labels, preds))
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold, best_score


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    sample_count = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        batch_size = int(batch_x.shape[0])
        running_loss += float(loss.item()) * batch_size
        sample_count += batch_size

    return float(running_loss / max(sample_count, 1))


def prepare_splits(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
    test_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    return x_train, x_test, y_train, y_test


def standardize_flat_features(
    x_reference: np.ndarray,
    *arrays: np.ndarray,
) -> tuple[np.ndarray, ...]:
    scaler = StandardScaler()
    scaler.fit(x_reference)
    return tuple(scaler.transform(array).astype(np.float32) for array in arrays)


def standardize_structured_features(
    x_reference: np.ndarray,
    *arrays: np.ndarray,
) -> tuple[np.ndarray, ...]:
    if x_reference.ndim != 3 or any(array.ndim != 3 for array in arrays):
        raise ValueError(
            "Structured standardization expects [N, C, T] arrays for both the reference split "
            f"and transformed arrays, got reference {x_reference.shape} and "
            f"{[array.shape for array in arrays]}"
        )
    mean = x_reference.mean(axis=(0, 2), keepdims=True)
    std = x_reference.std(axis=(0, 2), keepdims=True)
    std = np.maximum(std, 1e-6)
    return tuple(((array - mean) / std).astype(np.float32) for array in arrays)


def build_model(
    args: argparse.Namespace,
    input_shape: tuple[int, ...],
) -> nn.Module:
    hidden_dims = parse_hidden_dims(args.hidden_dims)
    if args.feature_set in {"raw", "codes", "unquantized_codes"}:
        conv_channels = parse_hidden_dims(args.conv_channels)
        if len(input_shape) != 2:
            raise ValueError(f"ConvSequenceClassifier expects [C, T] input shape, got {input_shape}")
        return ConvSequenceClassifier(
            input_channels=int(input_shape[0]),
            conv_channels=conv_channels,
            head_hidden_dims=hidden_dims,
            dropout=args.dropout,
        )

    input_dim = int(np.prod(input_shape))
    return BinaryMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
    )


def run_single_seed_experiment(
    seed: int,
    args: argparse.Namespace,
    features: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    print_stage(f"Preparing train/test split for seed={seed}")
    x_train, x_test, y_train, y_test = prepare_splits(
        features=features,
        labels=labels,
        seed=seed,
        test_size=args.test_size,
    )

    print_stage(f"Standardizing features for seed={seed}")
    # if x_train.ndim == 3:
    #     x_train, x_test = standardize_structured_features(x_train, x_train, x_test)
    # else:
    #     x_train, x_test = standardize_flat_features(x_train, x_train, x_test)

    train_loader = make_loader(x_train, y_train, batch_size=args.batch_size, shuffle=True)
    train_eval_loader = make_loader(x_train, y_train, batch_size=args.batch_size, shuffle=False)
    test_loader = make_loader(x_test, y_test, batch_size=args.batch_size, shuffle=False)

    print_stage(f"Building classifier for seed={seed}")
    model = build_model(args=args, input_shape=tuple(x_train.shape[1:])).to(device)

    num_positive = max(int(np.sum(y_train == 1)), 1)
    num_negative = max(int(np.sum(y_train == 0)), 1)
    pos_weight = torch.tensor([num_negative / num_positive], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_state = copy.deepcopy(model.state_dict())
    best_train_loss = float("inf")
    best_test_loss = float("inf")
    best_test_balanced_accuracy = float("-inf")

    print_stage(f"Training model for seed={seed}")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        test_metrics = evaluate_model(
            model=model,
            loader=test_loader,
            device=device,
            criterion=criterion,
            threshold=0.5,
        )
        scheduler.step()
        best_train_loss = min(best_train_loss, float(train_loss))

        current_test_loss = float(test_metrics["loss"])
        current_test_balanced_accuracy = float(test_metrics["balanced_accuracy"])
        improved = (
            current_test_balanced_accuracy > best_test_balanced_accuracy + 1e-6
            or (
                abs(current_test_balanced_accuracy - best_test_balanced_accuracy) <= 1e-6
                and current_test_loss < best_test_loss - 1e-6
            )
        )
        if improved:
            best_test_loss = current_test_loss
            best_test_balanced_accuracy = current_test_balanced_accuracy
            best_state = copy.deepcopy(model.state_dict())

        print(
            (
                f"[Epoch {epoch:03d}] seed={seed} "
                f"train_loss={train_loss:.6f} "
                f"test_loss={current_test_loss:.6f} "
                f"test_bal_acc={current_test_balanced_accuracy:.6f} "
                f"lr={scheduler.get_last_lr()[0]:.6e} "
                f"train_pos={int(np.sum(y_train == 1))} "
                f"train_neg={int(np.sum(y_train == 0))}"
            ),
            flush=True,
        )

    model.load_state_dict(best_state)

    selected_threshold = float(args.threshold)
    train_threshold_score = float("nan")
    if args.threshold_mode == "train_optimal":
        train_probs, train_labels = collect_probabilities(
            model=model,
            loader=train_eval_loader,
            device=device,
        )
        selected_threshold, train_threshold_score = find_best_threshold(train_probs, train_labels)

    print_stage(f"Evaluating trained model on test split for seed={seed}")
    train_metrics = evaluate_model(
        model=model,
        loader=train_eval_loader,
        device=device,
        criterion=criterion,
        threshold=selected_threshold,
    )
    test_metrics = evaluate_model(
        model=model,
        loader=test_loader,
        device=device,
        criterion=criterion,
        threshold=selected_threshold,
    )

    return {
        "seed": seed,
        "train_samples": int(x_train.shape[0]),
        "test_samples": int(x_test.shape[0]),
        "best_train_loss": float(best_train_loss),
        "selected_threshold": float(selected_threshold),
        "train_threshold_score": float(train_threshold_score),
        "train_accuracy": float(train_metrics["accuracy"]),
        "train_roc_auc": float(train_metrics["roc_auc"]),
        "accuracy": float(test_metrics["accuracy"]),
        "balanced_accuracy": float(test_metrics["balanced_accuracy"]),
        "f1": float(test_metrics["f1"]),
        "roc_auc": float(test_metrics["roc_auc"]),
        "confusion_matrix": test_metrics["confusion_matrix"],
        "classification_report": test_metrics["classification_report"],
        "test_score_mean_clean": float(test_metrics["score_mean_clean"]),
        "test_score_mean_anomaly": float(test_metrics["score_mean_anomaly"]),
        "test_logit_mean_clean": float(test_metrics["logit_mean_clean"]),
        "test_logit_mean_anomaly": float(test_metrics["logit_mean_anomaly"]),
    }


def main() -> None:
    print_stage("Parsing arguments")
    parser = argparse.ArgumentParser(
        description=(
            "Classify clean vs OFDM anomaly IQ windows with a PyTorch classifier using either raw IQ samples "
            "or foundation-model latent features."
        )
    )
    parser.add_argument(
        "--clean-dir",
        type=str,
        default="anomaly_detection/data/June_downsampling",
        help="Folder containing clean split files.",
    )
    parser.add_argument(
        "--anomaly-dir",
        type=str,
        default="anomaly_detection/data/June_downsampling",
        help="Folder containing generated anomaly split files.",
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        default="unquantized_codes",
        choices=["raw", "codes", "unquantized_codes", "code_ids"],
        help="Feature source for classification.",
    )
    parser.add_argument(
        "--code-id-encoding",
        type=str,
        default="sequence",
        choices=["histogram", "sequence"],
        help="How to convert code_ids into classifier inputs.",
    )
    parser.add_argument(
        "--foundation-model-path",
        type=str,
        # default="anomaly_detection/saved_models/down_sampling_160_anomaly/checkpoints/final_model.pth",
            # default="saved_models/model_stft(mse)_mse/checkpoints/model_epoch_3500.pth",
            default="saved_models/CD64_CW512_CF4_BS512_ITR20000_seed13_maskratio0.2/checkpoints/model_epoch_5500.pth",
        help="Path to the pretrained foundation/VQ-VAE model used to generate codes when needed.",
    )
    parser.add_argument(
        "--compression-factor",
        type=int,
        default=None,
        help="Override the model compression factor when generating foundation features.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=0,
        help="Optional cap per split and per class for faster experiments.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.99,
        help="Fraction of the labeled dataset reserved for testing.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Number of consecutive random seeds to evaluate, starting from --seed.",
    )
    parser.add_argument(
        "--hidden-dims",
        type=str,
        default="256,128",
        help="Comma-separated hidden layer sizes for the classifier head.",
    )
    parser.add_argument(
        "--conv-channels",
        type=str,
        default="64,128,128",
        help="Comma-separated Conv1D channel sizes used for structured raw/code inputs.",
    )
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument(
        "--threshold-mode",
        type=str,
        default="fixed",
        choices=["fixed", "train_optimal"],
        help="Use the fixed threshold directly or tune it on the training split.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--save-json",
        type=str,
        default="",
        help="Optional path to save metrics JSON.",
    )
    args = parser.parse_args()

    if not 0.0 < args.test_size < 1.0:
        raise ValueError(f"--test-size must be in (0, 1), got {args.test_size}")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError(f"--dropout must be in [0, 1), got {args.dropout}")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError(f"--threshold must be in (0, 1), got {args.threshold}")
    if args.epochs <= 0:
        raise ValueError(f"--epochs must be positive, got {args.epochs}")

    print_stage("Setting random seed")
    set_seed(args.seed)
    max_samples = args.max_samples_per_split or None

    model = None
    compression_factor = None
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if args.feature_set != "raw":
        print_stage("Loading foundation model for latent feature generation")
        model = load_model(args.foundation_model_path, device)
        compression_factor = args.compression_factor
        if compression_factor is None:
            compression_factor = int(getattr(model, "compression_factor"))

    print_stage("Loading clean features")
    clean_splits, clean_paths, feature_source, clean_generated_cache_paths = resolve_feature_inputs(
        data_dir=args.clean_dir,
        feature_set=args.feature_set,
        anomaly=False,
        max_samples_per_split=max_samples,
        model=model,
        compression_factor=compression_factor,
        batch_size=args.batch_size,
        device=device,
    )
    print_stage("Loading anomaly features")
    anomaly_splits, anomaly_paths, _, anomaly_generated_cache_paths = resolve_feature_inputs(
        data_dir=args.anomaly_dir,
        feature_set=args.feature_set,
        anomaly=True,
        max_samples_per_split=max_samples,
        model=model,
        compression_factor=compression_factor,
        batch_size=args.batch_size,
        device=device,
    )

    print_stage("Combining and transforming features")
    clean_features_raw = combine_splits(clean_splits)
    anomaly_features_raw = combine_splits(anomaly_splits)
    if clean_features_raw.shape[1:] != anomaly_features_raw.shape[1:]:
        raise ValueError(
            "Clean and anomaly data do not share the same per-sample shape. "
            f"Clean shape: {clean_features_raw.shape[1:]}; anomaly shape: {anomaly_features_raw.shape[1:]}."
        )
    input_feature_shape = list(clean_features_raw.shape[1:])
    input_feature_dim = summarize_feature_dim(clean_features_raw)

    clean_features, anomaly_features, classifier_feature_shape, classifier_feature_dim, feature_representation = (
        prepare_features_for_model(
            clean_features=clean_features_raw,
            anomaly_features=anomaly_features_raw,
            feature_set=args.feature_set,
            code_id_encoding=args.code_id_encoding,
        )
    )
    features, labels = build_labeled_split(clean_features, anomaly_features)

    print_stage("Running multi-seed evaluation")
    seeds = [args.seed + idx for idx in range(args.num_seeds)]
    per_seed_metrics = []
    for seed in seeds:
        set_seed(seed)
        per_seed_metrics.append(
            run_single_seed_experiment(
                seed=seed,
                args=args,
                features=features,
                labels=labels,
                device=device,
            )
        )

    print_stage("Computing averaged metrics")
    averaged_metrics = {
        "train_samples": average_numeric_structure([item["train_samples"] for item in per_seed_metrics]),
        "test_samples": average_numeric_structure([item["test_samples"] for item in per_seed_metrics]),
        "best_train_loss": average_numeric_structure([item["best_train_loss"] for item in per_seed_metrics]),
        "selected_threshold": average_numeric_structure([item["selected_threshold"] for item in per_seed_metrics]),
        "train_threshold_score": average_numeric_structure([item["train_threshold_score"] for item in per_seed_metrics]),
        "train_accuracy": average_numeric_structure([item["train_accuracy"] for item in per_seed_metrics]),
        "train_roc_auc": average_numeric_structure([item["train_roc_auc"] for item in per_seed_metrics]),
        "accuracy": average_numeric_structure([item["accuracy"] for item in per_seed_metrics]),
        "balanced_accuracy": average_numeric_structure([item["balanced_accuracy"] for item in per_seed_metrics]),
        "f1": average_numeric_structure([item["f1"] for item in per_seed_metrics]),
        "roc_auc": average_numeric_structure([item["roc_auc"] for item in per_seed_metrics]),
        "confusion_matrix": average_numeric_structure([item["confusion_matrix"] for item in per_seed_metrics]),
        "classification_report": average_numeric_structure([item["classification_report"] for item in per_seed_metrics]),
        "test_score_mean_clean": average_numeric_structure([item["test_score_mean_clean"] for item in per_seed_metrics]),
        "test_score_mean_anomaly": average_numeric_structure(
            [item["test_score_mean_anomaly"] for item in per_seed_metrics]
        ),
        "test_logit_mean_clean": average_numeric_structure([item["test_logit_mean_clean"] for item in per_seed_metrics]),
        "test_logit_mean_anomaly": average_numeric_structure(
            [item["test_logit_mean_anomaly"] for item in per_seed_metrics]
        ),
    }

    metrics: dict[str, Any] = {
        "clean_dir": args.clean_dir,
        "anomaly_dir": args.anomaly_dir,
        "clean_paths": clean_paths,
        "anomaly_paths": anomaly_paths,
        "feature_set": args.feature_set,
        "feature_representation": feature_representation,
        "feature_source": feature_source,
        "code_id_encoding": args.code_id_encoding if args.feature_set == "code_ids" else "not_applicable",
        "quantized_source": feature_source if args.feature_set != "raw" else "not_applicable",
        "generated_cache_paths": clean_generated_cache_paths + anomaly_generated_cache_paths,
        "foundation_model_path": args.foundation_model_path if args.feature_set != "raw" else "",
        "compression_factor": compression_factor if compression_factor is not None else None,
        "batch_size": args.batch_size,
        "device": str(device),
        "hidden_dims": parse_hidden_dims(args.hidden_dims),
        "conv_channels": (
            parse_hidden_dims(args.conv_channels)
            if args.feature_set in {"raw", "codes", "unquantized_codes"}
            else []
        ),
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "threshold_mode": args.threshold_mode,
        "threshold": args.threshold,
        "seed": args.seed,
        "num_seeds": args.num_seeds,
        "seeds": seeds,
        "test_size": args.test_size,
        "num_clean_samples": int(clean_features_raw.shape[0]),
        "num_anomaly_samples": int(anomaly_features_raw.shape[0]),
        "num_total_samples": int(features.shape[0]),
        "train_samples": averaged_metrics["train_samples"],
        "test_samples": averaged_metrics["test_samples"],
        "input_feature_shape": input_feature_shape,
        "input_feature_dim": input_feature_dim,
        "classifier_feature_shape": classifier_feature_shape,
        "classifier_feature_dim": classifier_feature_dim,
        "best_train_loss": averaged_metrics["best_train_loss"],
        "selected_threshold": averaged_metrics["selected_threshold"],
        "train_threshold_score": averaged_metrics["train_threshold_score"],
        "train_accuracy": averaged_metrics["train_accuracy"],
        "train_roc_auc": averaged_metrics["train_roc_auc"],
        "accuracy": averaged_metrics["accuracy"],
        "balanced_accuracy": averaged_metrics["balanced_accuracy"],
        "f1": averaged_metrics["f1"],
        "roc_auc": averaged_metrics["roc_auc"],
        "confusion_matrix": averaged_metrics["confusion_matrix"],
        "classification_report": averaged_metrics["classification_report"],
        "test_score_mean_clean": averaged_metrics["test_score_mean_clean"],
        "test_score_mean_anomaly": averaged_metrics["test_score_mean_anomaly"],
        "test_logit_mean_clean": averaged_metrics["test_logit_mean_clean"],
        "test_logit_mean_anomaly": averaged_metrics["test_logit_mean_anomaly"],
        "per_seed_metrics": per_seed_metrics,
    }

    print_stage("Printing results")
    print("=== MLP Clean vs OFDM Anomaly Classification ===")
    print(f"clean paths: {', '.join(clean_paths)}")
    print(f"anomaly paths: {', '.join(anomaly_paths)}")
    print(f"feature set: {metrics['feature_set']}")
    print(f"feature representation: {metrics['feature_representation']}")
    print(f"feature source: {metrics['feature_source']}")
    print(f"input feature shape per sample: {tuple(metrics['input_feature_shape'])}")
    print(f"input feature dim: {metrics['input_feature_dim']}")
    print(f"classifier feature shape per sample: {tuple(metrics['classifier_feature_shape'])}")
    print(f"classifier feature dim: {metrics['classifier_feature_dim']}")
    if args.feature_set != "raw":
        print(f"quantized source: {metrics['quantized_source']}")
        if metrics["compression_factor"] is not None:
            print(f"compression factor: {metrics['compression_factor']}")
        print(f"foundation model: {metrics['foundation_model_path']}")
        if args.feature_set == "code_ids":
            print(f"code_id encoding: {metrics['code_id_encoding']}")
    if metrics["conv_channels"]:
        print(f"conv channels: {metrics['conv_channels']}")
    print(f"hidden dims: {metrics['hidden_dims']}")
    print(f"dropout: {metrics['dropout']}")
    print(f"learning rate: {metrics['learning_rate']}")
    print(f"weight decay: {metrics['weight_decay']}")
    print(f"threshold mode: {metrics['threshold_mode']}")
    print(f"threshold: {metrics['threshold']}")
    print(f"clean samples: {metrics['num_clean_samples']}")
    print(f"anomaly samples: {metrics['num_anomaly_samples']}")
    print(f"seeds: {metrics['seeds']}")
    print(f"train samples (avg): {metrics['train_samples']:.2f}")
    print(f"test samples (avg): {metrics['test_samples']:.2f}")
    print(f"best train loss (avg): {metrics['best_train_loss']:.6f}")
    print(f"selected threshold (avg): {metrics['selected_threshold']:.6f}")
    print(f"train threshold score (avg): {metrics['train_threshold_score']:.6f}")
    print(f"train accuracy (avg): {metrics['train_accuracy']:.6f}")
    print(f"train roc_auc (avg): {metrics['train_roc_auc']:.6f}")
    print(f"test accuracy (avg): {metrics['accuracy']:.6f}")
    print(f"test balanced_accuracy (avg): {metrics['balanced_accuracy']:.6f}")
    print(f"test f1 (avg): {metrics['f1']:.6f}")
    print(f"test roc_auc (avg): {metrics['roc_auc']:.6f}")
    print(f"mean test probability clean (avg): {metrics['test_score_mean_clean']:.6f}")
    print(f"mean test probability anomaly (avg): {metrics['test_score_mean_anomaly']:.6f}")
    print(f"mean test logit clean (avg): {metrics['test_logit_mean_clean']:.6f}")
    print(f"mean test logit anomaly (avg): {metrics['test_logit_mean_anomaly']:.6f}")
    print("confusion_matrix (avg):")
    print(np.array(metrics["confusion_matrix"]))

    if args.save_json:
        print_stage("Saving metrics JSON")
        output_dir = os.path.dirname(args.save_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to: {args.save_json}")


if __name__ == "__main__":
    main()
