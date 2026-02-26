import json
import os
import pickle
import random
import time
from itertools import product

import numpy as np
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def load_npz_array(npz_path, key):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing file: {npz_path}")
    with np.load(npz_path, allow_pickle=True) as data:
        if key not in data:
            raise KeyError(f"Key '{key}' not found in {npz_path}. Available keys: {list(data.keys())}")
        return data[key]


def load_label_array(label_path, label_key):
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Missing file: {label_path}")

    ext = os.path.splitext(label_path)[1].lower()
    if ext == ".npy":
        return np.load(label_path, allow_pickle=True)

    if ext == ".npz":
        with np.load(label_path, allow_pickle=True) as data:
            if label_key in data:
                return data[label_key]
            if len(data.files) == 1:
                fallback_key = data.files[0]
                print(
                    f"[Info] Label key '{label_key}' not found in {label_path}; "
                    f"using only available key '{fallback_key}'."
                )
                return data[fallback_key]
            raise KeyError(
                f"Key '{label_key}' not found in {label_path}. Available keys: {list(data.keys())}"
            )

    raise ValueError(
        f"Unsupported label file extension '{ext}' for {label_path}. Use .npy or .npz."
    )


def load_feature_array(path, preferred_key):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path, allow_pickle=True)
        # Only raw RevIN npy should be reformatted.
        if os.path.basename(path).endswith("revin_x.npy") and arr.ndim == 3:
            per_sample = int(np.prod(arr.shape[1:]))
            expected = 2 * 10 * 1024
            if per_sample != expected:
                raise ValueError(
                    f"Expected per-sample size {expected} for {path}, got {per_sample} with shape {arr.shape}"
                )
            n = arr.shape[0]
            arr = arr.reshape(n, 2, 10, 1024).transpose(0, 2, 1, 3)
        return arr, None

    if ext == ".npz":
        with np.load(path, allow_pickle=True) as data:
            if preferred_key in data:
                return data[preferred_key], preferred_key
            if preferred_key == "codes" and "code_ids" in data:
                return data["code_ids"], "code_ids"
            raise KeyError(
                f"Key '{preferred_key}' not found in {path}. Available keys: {list(data.keys())}"
            )

    raise ValueError(
        f"Unsupported feature file extension '{ext}' for {path}. Use .npy or .npz."
    )


def load_split_arrays(base_path, code_suffix, label_suffix, code_key, label_key, prn):
    split_data = {}
    all_labels = []

    for split in ("train", "val", "test"):
        code_path = os.path.join(base_path, f"{split}_{code_suffix}")
        label_path = os.path.join(base_path, f"{split}_{label_suffix}")

        codes, resolved_key = load_feature_array(code_path, code_key)
        if resolved_key is not None and resolved_key != code_key:
            print(f"[Info] Using fallback key '{resolved_key}' for {code_path}")
        labels_arr = load_label_array(label_path, label_key)
        if labels_arr.ndim < 2:
            raise ValueError(
                f"Expected labels to have shape [N, num_prn] for PRN selection, got {labels_arr.shape} in {label_path}"
            )
        prn_idx = int(prn) - 1
        if prn_idx < 0 or prn_idx >= labels_arr.shape[1]:
            raise IndexError(
                f"target_prn={prn} is out of bounds for labels with shape {labels_arr.shape} in {label_path}"
            )
        labels = labels_arr[:, prn_idx]

        if len(codes) != len(labels):
            raise ValueError(
                f"Sample count mismatch in split '{split}': codes={len(codes)} labels={len(labels)}"
            )

        split_data[split] = (codes, labels)
        all_labels.append(labels)

    y_raw_all = np.concatenate(all_labels, axis=0)
    unique = sorted(list(set(y_raw_all.tolist())))
    label_map = {int(v): i for i, v in enumerate(unique)}

    out = {}
    for split, (x, y_raw) in split_data.items():
        y = np.array([label_map[int(v)] for v in y_raw], dtype=np.int64)
        out[split] = (x, y)
    return out, label_map


def normalize_splits_from_train(split_arrays):
    x_train, _ = split_arrays["train"]
    if x_train.ndim == 4:
        mean = x_train.mean(axis=(0, 1, 3), keepdims=True).astype(np.float32)
        std = x_train.std(axis=(0, 1, 3), keepdims=True).astype(np.float32)
    elif x_train.ndim == 3:
        mean = x_train.mean(axis=(0, 1), keepdims=True).astype(np.float32)
        std = x_train.std(axis=(0, 1), keepdims=True).astype(np.float32)
    else:
        mean = x_train.mean(axis=0, keepdims=True).astype(np.float32)
        std = x_train.std(axis=0, keepdims=True).astype(np.float32)

    std = np.maximum(std, 1e-6)
    normalized = {}
    for split, (x, y) in split_arrays.items():
        normalized[split] = ((x.astype(np.float32) - mean) / std, y)
    return normalized


def normalize_pair_from_train(x_train, x_val):
    if x_train.ndim == 4:
        mean = x_train.mean(axis=(0, 1, 3), keepdims=True).astype(np.float32)
        std = x_train.std(axis=(0, 1, 3), keepdims=True).astype(np.float32)
    elif x_train.ndim == 3:
        mean = x_train.mean(axis=(0, 1), keepdims=True).astype(np.float32)
        std = x_train.std(axis=(0, 1), keepdims=True).astype(np.float32)
    else:
        mean = x_train.mean(axis=0, keepdims=True).astype(np.float32)
        std = x_train.std(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(std, 1e-6)
    x_train_n = (x_train.astype(np.float32) - mean) / std
    x_val_n = (x_val.astype(np.float32) - mean) / std
    return x_train_n, x_val_n


def build_fixed_split_arrays(split_arrays, normalize=True):
    if normalize:
        split_arrays = normalize_splits_from_train(split_arrays)
    x_train, y_train = split_arrays["train"]
    x_val, y_val = split_arrays["val"]
    split_counts = {
        "train": len(x_train),
        "val": len(x_val),
        "total": len(x_train) + len(x_val),
    }
    return x_train, y_train, x_val, y_val, split_counts


def build_stratified_resplit_arrays(split_arrays, seed, train_ratio, val_ratio, normalize=True):
    x_all = np.concatenate([split_arrays["train"][0], split_arrays["val"][0], split_arrays["test"][0]], axis=0)
    y_all = np.concatenate([split_arrays["train"][1], split_arrays["val"][1], split_arrays["test"][1]], axis=0)

    ratios = np.array([train_ratio, val_ratio], dtype=np.float64)
    if np.any(ratios < 0):
        raise ValueError("Split ratios must be non-negative")
    if ratios.sum() <= 0:
        raise ValueError("Split ratios must sum to > 0")
    ratios = ratios / ratios.sum()
    val_size = float(ratios[1])

    x_train, x_val, y_train, y_val = train_test_split(
        x_all, y_all, test_size=val_size, random_state=seed, stratify=y_all
    )
    if normalize:
        x_train, x_val = normalize_pair_from_train(x_train, x_val)
    split_counts = {
        "train": len(x_train),
        "val": len(x_val),
        "total": len(x_train) + len(x_val),
    }
    return x_train, y_train, x_val, y_val, split_counts


def extract_code_aware_features(x, vocab_size=None, bigram_bins=128):
    x = np.asarray(x)
    if x.ndim < 2:
        raise ValueError(f"code_aware expects token arrays with at least 2 dims [N, ...], got {x.shape}")

    tokens = x.reshape(x.shape[0], -1).astype(np.int64)
    n, t = tokens.shape
    if t < 2:
        raise ValueError(f"code_aware expects at least 2 tokens per sample, got shape {x.shape}")

    token_min = int(tokens.min())
    if token_min < 0:
        tokens = tokens - token_min

    if vocab_size is None:
        vocab_size = int(tokens.max()) + 1
    vocab_size = max(2, int(vocab_size))
    bigram_bins = max(8, int(bigram_bins))

    hist = np.zeros((n, vocab_size), dtype=np.float32)
    uniq_ratio = np.zeros((n, 1), dtype=np.float32)
    entropy = np.zeros((n, 1), dtype=np.float32)
    change_ratio = np.zeros((n, 1), dtype=np.float32)
    bigram_hist = np.zeros((n, bigram_bins), dtype=np.float32)

    for i in range(n):
        row = np.clip(tokens[i], 0, vocab_size - 1)
        counts = np.bincount(row, minlength=vocab_size).astype(np.float32)
        probs = counts / max(float(t), 1.0)
        hist[i] = probs

        nz = probs[probs > 0]
        uniq_ratio[i, 0] = float(len(nz)) / float(t)
        entropy[i, 0] = float(-(nz * np.log(nz + 1e-12)).sum())
        change_ratio[i, 0] = float((row[1:] != row[:-1]).mean())

        left = row[:-1].astype(np.int64)
        right = row[1:].astype(np.int64)
        pair_hash = (left * 1315423911 + right * 2654435761) % bigram_bins
        pair_counts = np.bincount(pair_hash, minlength=bigram_bins).astype(np.float32)
        bigram_hist[i] = pair_counts / max(float(t - 1), 1.0)

    return np.concatenate([hist, bigram_hist, uniq_ratio, entropy, change_ratio], axis=1)


def extract_features(x, feature_mode, code_feature_cfg=None):
    x_raw = np.asarray(x)
    x = x_raw.astype(np.float32)

    if feature_mode == "code_aware":
        cfg = code_feature_cfg or {}
        if not np.issubdtype(x_raw.dtype, np.integer):
            raise ValueError(
                f"feature_mode='code_aware' requires discrete integer token IDs, got dtype={x_raw.dtype}"
            )
        vocab_size = cfg.get("code_vocab_size", None)
        if vocab_size is None and np.issubdtype(x_raw.dtype, np.integer):
            vocab_size = int(np.max(x_raw)) + 1
        bigram_bins = int(cfg.get("code_bigram_bins", 128))
        return extract_code_aware_features(x_raw, vocab_size=vocab_size, bigram_bins=bigram_bins)

    if feature_mode == "flat":
        return x.reshape(x.shape[0], -1)

    if feature_mode == "pooled":
        if x.ndim == 4:
            # For [N, P, C, T], pool over pieces and time to keep channel-level features.
            return x.mean(axis=(1, 3)).reshape(x.shape[0], -1)
        if x.ndim == 3:
            return x.mean(axis=1)
        return x.reshape(x.shape[0], -1)

    if feature_mode == "pooled_stats":
        if x.ndim == 4:
            # For [N, P, C, T], summarize over pieces and time.
            mean = x.mean(axis=(1, 3)).reshape(x.shape[0], -1)
            std = x.std(axis=(1, 3)).reshape(x.shape[0], -1)
            maxv = x.max(axis=(1, 3)).reshape(x.shape[0], -1)
            minv = x.min(axis=(1, 3)).reshape(x.shape[0], -1)
            return np.concatenate([mean, std, maxv, minv], axis=1)
        if x.ndim == 3:
            mean = x.mean(axis=1)
            std = x.std(axis=1)
            maxv = x.max(axis=1)
            minv = x.min(axis=1)
            return np.concatenate([mean, std, maxv, minv], axis=1)
        flat = x.reshape(x.shape[0], -1)
        return np.concatenate(
            [flat, np.square(flat), np.abs(flat)],
            axis=1,
        )

    if feature_mode == "pooled_rich":
        if x.ndim == 4:
            # For [N, P, C, T], summarize over pieces and time.
            mean = x.mean(axis=(1, 3)).reshape(x.shape[0], -1)
            std = x.std(axis=(1, 3)).reshape(x.shape[0], -1)
            maxv = x.max(axis=(1, 3)).reshape(x.shape[0], -1)
            minv = x.min(axis=(1, 3)).reshape(x.shape[0], -1)
            q25 = np.quantile(x, 0.25, axis=(1, 3)).reshape(x.shape[0], -1)
            q75 = np.quantile(x, 0.75, axis=(1, 3)).reshape(x.shape[0], -1)
            iqr = q75 - q25
            return np.concatenate([mean, std, maxv, minv, q25, q75, iqr], axis=1)
        if x.ndim == 3:
            mean = x.mean(axis=1)
            std = x.std(axis=1)
            maxv = x.max(axis=1)
            minv = x.min(axis=1)
            q25 = np.quantile(x, 0.25, axis=1)
            q75 = np.quantile(x, 0.75, axis=1)
            iqr = q75 - q25
            return np.concatenate([mean, std, maxv, minv, q25, q75, iqr], axis=1)
        flat = x.reshape(x.shape[0], -1)
        return np.concatenate([flat, np.square(flat), np.abs(flat)], axis=1)

    raise ValueError(
        f"Unknown feature_mode '{feature_mode}'. Use one of: flat, pooled, pooled_stats, pooled_rich, code_aware"
    )


def run_eval_xgb(model, x, y):
    y_pred = model.predict(x)
    y_prob = model.predict_proba(x)[:, 1]
    acc = accuracy_score(y, y_pred)
    try:
        auc = roc_auc_score(y, y_prob)
    except ValueError:
        auc = float("nan")
    return float(acc), float(auc)


def eval_from_probs(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(np.int64)
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    return float(acc), float(auc)


def build_search_space(cfg):
    max_depths = list(cfg.get("search_max_depths", [2, 3, 4]))
    min_child_weights = list(cfg.get("search_min_child_weights", [1, 2, 4, 6]))
    subsamples = list(cfg.get("search_subsamples", [0.5, 0.7, 0.9]))
    colsample_bytree = list(cfg.get("search_colsample_bytree", [0.5, 0.7, 0.9]))
    reg_alphas = list(cfg.get("search_reg_alphas", [0.0, 0.5, 1.0, 2.0]))
    reg_lambdas = list(cfg.get("search_reg_lambdas", [1.0, 2.0, 5.0, 10.0]))
    learning_rates = list(cfg.get("search_learning_rates", [0.01, 0.03, 0.05, 0.08]))
    gammas = list(cfg.get("search_gammas", [0.0, 0.1, 0.3]))
    max_delta_steps = list(cfg.get("search_max_delta_steps", [0.0, 1.0, 2.0]))
    boosters = list(cfg.get("search_boosters", ["gbtree"]))

    all_candidates = list(
        product(
            max_depths,
            min_child_weights,
            subsamples,
            colsample_bytree,
            reg_alphas,
            reg_lambdas,
            learning_rates,
            gammas,
            max_delta_steps,
            boosters,
        )
    )
    return all_candidates


def _safe_metric(v, default=-1e12):
    if v is None:
        return float(default)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return float(default)
    if np.isnan(v):
        return float(default)
    return float(v)


def select_candidates_val_then_train(all_results, code_cfg):
    valid = [r for r in all_results if not np.isnan(r.get("val_auc", np.nan))]
    if len(valid) == 0:
        return []

    for r in valid:
        r["train_score"] = 0.5 * _safe_metric(r.get("train_auc")) + 0.5 * _safe_metric(r.get("train_acc"))
        r["val_score"] = 0.8 * _safe_metric(r.get("val_auc")) + 0.2 * _safe_metric(r.get("val_acc"))

    prefilter_ratio = float(code_cfg.get("val_prefilter_ratio", 0.4))
    prefilter_ratio = min(max(prefilter_ratio, 0.0), 1.0)
    prefilter_k_cfg = code_cfg.get("val_prefilter_k", None)

    valid_sorted_by_val = sorted(
        valid,
        key=lambda r: (
            r["val_score"],
            _safe_metric(r.get("val_auc")),
            _safe_metric(r.get("val_acc")),
            r["train_score"],
            _safe_metric(r.get("train_auc")),
            _safe_metric(r.get("train_acc")),
        ),
        reverse=True,
    )

    if prefilter_k_cfg is None:
        keep_val_k = max(1, int(np.ceil(len(valid_sorted_by_val) * prefilter_ratio)))
    else:
        keep_val_k = max(1, int(prefilter_k_cfg))
    keep_val_k = min(keep_val_k, len(valid_sorted_by_val))
    val_kept = valid_sorted_by_val[:keep_val_k]

    min_train_auc = code_cfg.get("min_train_auc", None)
    min_train_acc = code_cfg.get("min_train_acc", None)
    if min_train_auc is not None or min_train_acc is not None:
        val_kept_thresholded = []
        for r in val_kept:
            ok_auc = True if min_train_auc is None else _safe_metric(r.get("train_auc")) >= float(min_train_auc)
            ok_acc = True if min_train_acc is None else _safe_metric(r.get("train_acc")) >= float(min_train_acc)
            if ok_auc and ok_acc:
                val_kept_thresholded.append(r)
        if len(val_kept_thresholded) > 0:
            val_kept = val_kept_thresholded

    sorted_final = sorted(
        val_kept,
        key=lambda r: (
            r["val_score"],
            _safe_metric(r.get("val_auc")),
            _safe_metric(r.get("val_acc")),
            r["train_score"],
            _safe_metric(r.get("train_auc")),
            _safe_metric(r.get("train_acc")),
        ),
        reverse=True,
    )
    return sorted_final


def resolve_code_key(code_suffix, configured_code_key):
    ext = os.path.splitext(str(code_suffix))[1].lower()
    configured = str(configured_code_key).lower()
    if ext == ".npy":
        if configured != "codes":
            print(
                f"[Warn] code_suffix='{code_suffix}' is raw .npy; overriding code_key "
                f"from '{configured_code_key}' to 'codes'."
            )
        return "codes"
    if ext == ".npz":
        if configured != "code_ids":
            print(
                f"[Warn] code_suffix='{code_suffix}' is quantized .npz; overriding code_key "
                f"from '{configured_code_key}' to 'code_ids'."
            )
        return "code_ids"
    raise ValueError(
        f"Unsupported code_suffix extension '{ext}' for {code_suffix}. Use .npy (raw) or .npz (quantized)."
    )


def main():
    config = OmegaConf.load("quantized_code_xgboost_config.yaml")
    code_cfg = config.get("code_xgboost", config.get("code_classifier", {}))

    seed = int(config.get("seed", 13))
    np.random.seed(seed)
    random.seed(seed)

    base_path = code_cfg.get("base_path", "IQ_labeled_data/June_downsampling_10ms")
    code_suffix = code_cfg.get("code_suffix", "notrevin_x_quantized_codes.npz")
    label_suffix = code_cfg.get("label_suffix", "notrevin_x_binary_labels.npz")
    configured_code_key = code_cfg.get("code_key", "codes")
    code_key = resolve_code_key(code_suffix, configured_code_key)
    label_key = code_cfg.get("label_key", "binary_classes")
    target_prn = int(code_cfg.get("target_prn", 4))
    split_mode = str(code_cfg.get("split_mode", "stratified_resplit")).lower()
    train_ratio = float(code_cfg.get("train_ratio", 0.6))
    val_ratio = float(code_cfg.get("val_ratio", 0.4))
    feature_mode = str(code_cfg.get("feature_mode", "auto")).lower()
    feature_modes = code_cfg.get("feature_modes", None)
    save_path = code_cfg.get("save_path", "saved_models/code_classifier_xgboost")

    n_estimators = int(code_cfg.get("n_estimators", 800))
    early_stopping_rounds = int(code_cfg.get("early_stopping_rounds", 40))
    num_trials = int(code_cfg.get("num_trials", 24))
    max_search_seconds = float(code_cfg.get("max_search_seconds", 180.0))
    use_scale_pos_weight = bool(code_cfg.get("use_scale_pos_weight", True))
    normalize_inputs_default = str(code_key).lower() != "code_ids"
    normalize_inputs = bool(code_cfg.get("normalize_inputs", normalize_inputs_default))

    split_arrays, label_map = load_split_arrays(
        base_path=base_path,
        code_suffix=code_suffix,
        label_suffix=label_suffix,
        code_key=code_key,
        label_key=label_key,
        prn=target_prn,
    )

    if split_mode == "fixed":
        x_train, y_train, x_val, y_val, split_counts = build_fixed_split_arrays(
            split_arrays,
            normalize=normalize_inputs,
        )
    elif split_mode == "stratified_resplit":
        x_train, y_train, x_val, y_val, split_counts = build_stratified_resplit_arrays(
            split_arrays=split_arrays,
            seed=seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            normalize=normalize_inputs,
        )
    else:
        raise ValueError("Unknown split_mode. Use one of: fixed, stratified_resplit")

    if feature_modes is not None:
        feature_modes = [str(m).lower() for m in feature_modes]
    elif feature_mode == "auto":
        if str(code_key).lower() == "code_ids":
            feature_modes = ["code_aware", "pooled_rich", "pooled_stats"]
        else:
            feature_modes = ["pooled_rich", "pooled_stats"]
    else:
        feature_modes = [feature_mode]

    if "code_aware" in feature_modes and not np.issubdtype(x_train.dtype, np.integer):
        filtered_modes = [m for m in feature_modes if m != "code_aware"]
        print(
            "[Warn] Skipping feature_mode='code_aware' because input is not integer token IDs "
            f"(dtype={x_train.dtype})."
        )
        feature_modes = filtered_modes
    if len(feature_modes) == 0:
        raise ValueError(
            "No valid feature modes remain after compatibility checks. "
            "Use pooled/flat modes for raw or continuous inputs, or provide integer code_ids for code_aware."
        )

    if len(label_map) != 2:
        raise ValueError(f"XGBoost script currently supports binary classification. Found labels: {label_map}")

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = float(neg / max(pos, 1)) if use_scale_pos_weight else 1.0

    print(f"Loaded samples: {split_counts}")
    print(f"Feature modes: {feature_modes}")
    print(f"Class balance train: pos={pos}, neg={neg}, scale_pos_weight={scale_pos_weight:.4f}")

    candidates = build_search_space(code_cfg)
    rng = np.random.default_rng(seed)
    if num_trials > 0 and num_trials < len(candidates):
        chosen_indices = rng.choice(len(candidates), size=num_trials, replace=False)
        chosen = [candidates[int(i)] for i in chosen_indices]
    else:
        chosen = candidates

    best_auc = float("-inf")
    best_model = None
    best_params = None
    best_iteration = None
    search_start = time.time()
    all_results = []
    x_train_by_mode = {}
    x_val_by_mode = {}
    for mode in feature_modes:
        x_train_by_mode[mode] = extract_features(x_train, mode, code_cfg)
        x_val_by_mode[mode] = extract_features(x_val, mode, code_cfg)
        print(
            f"[Feature] {mode} -> train={x_train_by_mode[mode].shape} val={x_val_by_mode[mode].shape}"
        )

    total_trials = len(chosen) * len(feature_modes)
    trial_counter = 0
    for mode in feature_modes:
        x_train_f = x_train_by_mode[mode]
        x_val_f = x_val_by_mode[mode]

        for params in chosen:
            if (time.time() - search_start) >= max_search_seconds:
                print(
                    f"[Info] Reached max_search_seconds={max_search_seconds:.1f}. "
                    "Stopping search early."
                )
                break
            trial_counter += 1
            (
                max_depth,
                min_child_weight,
                subsample,
                colsample_bt,
                reg_alpha,
                reg_lambda,
                lr,
                gamma,
                max_delta_step,
                booster,
            ) = params

            if booster == "dart":
                rate_drop = float(code_cfg.get("dart_rate_drop", 0.1))
                skip_drop = float(code_cfg.get("dart_skip_drop", 0.3))
            else:
                rate_drop = 0.0
                skip_drop = 0.0

            model_kwargs = {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "n_estimators": n_estimators,
                "early_stopping_rounds": early_stopping_rounds,
                "learning_rate": float(lr),
                "max_depth": int(max_depth),
                "min_child_weight": float(min_child_weight),
                "subsample": float(subsample),
                "colsample_bytree": float(colsample_bt),
                "reg_alpha": float(reg_alpha),
                "reg_lambda": float(reg_lambda),
                "gamma": float(gamma),
                "max_delta_step": float(max_delta_step),
                "booster": str(booster),
                "random_state": seed + trial_counter,
                "n_jobs": int(code_cfg.get("n_jobs", -1)),
                "tree_method": str(code_cfg.get("tree_method", "hist")),
                "scale_pos_weight": scale_pos_weight,
            }
            if booster == "dart":
                model_kwargs["rate_drop"] = float(rate_drop)
                model_kwargs["skip_drop"] = float(skip_drop)

            model = XGBClassifier(
                **model_kwargs,
            )

            model.fit(
                x_train_f,
                y_train,
                eval_set=[(x_val_f, y_val)],
                verbose=False,
            )

            val_acc, val_auc = run_eval_xgb(model, x_val_f, y_val)
            train_acc, train_auc = run_eval_xgb(model, x_train_f, y_train)

            if hasattr(model, "best_iteration") and model.best_iteration is not None:
                best_iter_this = int(model.best_iteration)
            else:
                best_iter_this = int(n_estimators)

            print(
                f"Trial {trial_counter:03d}/{total_trials:03d} | mode={mode} booster={booster} | "
                f"train_acc={train_acc:.4f} train_auc={train_auc:.4f} | "
                f"val_acc={val_acc:.4f} val_auc={val_auc:.4f} | "
                f"best_iter={best_iter_this}"
            )

            if not np.isnan(val_auc) and val_auc > best_auc:
                best_auc = float(val_auc)
                best_model = model
                best_params = {
                    "max_depth": int(max_depth),
                    "min_child_weight": float(min_child_weight),
                    "subsample": float(subsample),
                    "colsample_bytree": float(colsample_bt),
                    "reg_alpha": float(reg_alpha),
                    "reg_lambda": float(reg_lambda),
                    "learning_rate": float(lr),
                    "gamma": float(gamma),
                    "max_delta_step": float(max_delta_step),
                    "booster": str(booster),
                    "rate_drop": float(rate_drop),
                    "skip_drop": float(skip_drop),
                    "n_estimators": int(n_estimators),
                    "scale_pos_weight": float(scale_pos_weight),
                    "feature_mode": mode,
                }
                best_iteration = best_iter_this
            all_results.append(
                {
                    "model": model,
                    "feature_mode": mode,
                    "train_prob": model.predict_proba(x_train_f)[:, 1],
                    "val_prob": model.predict_proba(x_val_f)[:, 1],
                    "train_acc": train_acc,
                    "train_auc": train_auc,
                    "val_acc": val_acc,
                    "val_auc": val_auc,
                    "best_iter": best_iter_this,
                    "params": {
                        "max_depth": int(max_depth),
                        "min_child_weight": float(min_child_weight),
                        "subsample": float(subsample),
                        "colsample_bytree": float(colsample_bt),
                        "reg_alpha": float(reg_alpha),
                        "reg_lambda": float(reg_lambda),
                        "learning_rate": float(lr),
                        "gamma": float(gamma),
                        "max_delta_step": float(max_delta_step),
                        "booster": str(booster),
                        "n_estimators": int(n_estimators),
                    },
                }
            )
        if (time.time() - search_start) >= max_search_seconds:
            break

    if best_model is None or len(all_results) == 0:
        raise RuntimeError("No valid model selected. Please check labels and data.")

    all_results = select_candidates_val_then_train(all_results, code_cfg)
    if len(all_results) == 0:
        raise RuntimeError("No valid model after val/train selection. Please check labels and data.")

    print(
        f"[Select] val-first prefilter kept {len(all_results)} candidates before ensemble top-k clip."
    )

    top_k = int(code_cfg.get("ensemble_top_k", 12))
    all_results = all_results[:max(1, top_k)]

    ensemble_selected = [all_results[0]]
    ens_train_prob = all_results[0]["train_prob"].copy()
    ens_val_prob = all_results[0]["val_prob"].copy()
    _, ens_val_auc = eval_from_probs(y_val, ens_val_prob)

    for cand in all_results[1:]:
        new_train_prob = (ens_train_prob * len(ensemble_selected) + cand["train_prob"]) / (
            len(ensemble_selected) + 1
        )
        new_val_prob = (ens_val_prob * len(ensemble_selected) + cand["val_prob"]) / (
            len(ensemble_selected) + 1
        )
        _, new_val_auc = eval_from_probs(y_val, new_val_prob)
        if np.isnan(new_val_auc):
            continue
        if new_val_auc > ens_val_auc + 1e-6:
            ensemble_selected.append(cand)
            ens_train_prob = new_train_prob
            ens_val_prob = new_val_prob
            ens_val_auc = new_val_auc

    train_acc, train_auc = eval_from_probs(y_train, ens_train_prob)
    val_acc, val_auc = eval_from_probs(y_val, ens_val_prob)
    best_iteration = int(round(np.mean([m["best_iter"] for m in ensemble_selected])))
    best_params = {
        "type": "greedy_average_ensemble",
        "num_members": len(ensemble_selected),
        "feature_modes": sorted(list(set(m["feature_mode"] for m in ensemble_selected))),
        "members": [
            {
                "feature_mode": m["feature_mode"],
                "val_auc": m["val_auc"],
                "train_auc": m["train_auc"],
                "params": m["params"],
            }
            for m in ensemble_selected
        ],
    }

    print(
        f"Best final | train_acc={train_acc:.4f} train_auc={train_auc:.4f} | "
        f"val_acc={val_acc:.4f} val_auc={val_auc:.4f} | best_iter={best_iteration}"
    )

    os.makedirs(save_path, exist_ok=True)
    model_file = os.path.join(save_path, "code_xgboost.pkl")
    meta_file = os.path.join(save_path, "metadata_xgboost.json")

    model_bundle = {
        "model_type": "xgboost_greedy_ensemble",
        "members": [
            {"feature_mode": m["feature_mode"], "model": m["model"]}
            for m in ensemble_selected
        ],
    }

    with open(model_file, "wb") as f:
        pickle.dump(model_bundle, f)

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "label_map": label_map,
                "split_counts": split_counts,
                "num_classes": len(label_map),
                "target_prn": target_prn,
                "seed": seed,
                "train_metrics": {"acc": train_acc, "auc": train_auc},
                "val_metrics": {"acc": val_acc, "auc": val_auc},
                "best_iteration": best_iteration,
                "best_params": best_params,
                "config": {
                    "base_path": base_path,
                    "code_suffix": code_suffix,
                    "label_suffix": label_suffix,
                    "code_key": code_key,
                    "configured_code_key": configured_code_key,
                    "label_key": label_key,
                    "split_mode": split_mode,
                    "train_ratio": train_ratio,
                    "val_ratio": val_ratio,
                    "feature_mode": feature_mode,
                    "num_trials": num_trials,
                    "n_estimators": n_estimators,
                    "early_stopping_rounds": early_stopping_rounds,
                    "max_search_seconds": max_search_seconds,
                    "use_scale_pos_weight": use_scale_pos_weight,
                    "normalize_inputs": normalize_inputs,
                },
            },
            f,
            indent=2,
        )

    print(f"Saved model to: {model_file}")
    print(f"Saved metadata to: {meta_file}")


if __name__ == "__main__":
    main()
