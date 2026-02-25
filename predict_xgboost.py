import argparse
import os
import pickle

import numpy as np
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, roc_auc_score


def load_npz_array(npz_path, key):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing file: {npz_path}")
    with np.load(npz_path, allow_pickle=True) as data:
        if key not in data:
            raise KeyError(f"Key '{key}' not found in {npz_path}. Available keys: {list(data.keys())}")
        return data[key]


def load_code_array(npz_path, preferred_key):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing file: {npz_path}")
    with np.load(npz_path, allow_pickle=True) as data:
        if preferred_key in data:
            return data[preferred_key], preferred_key
        if preferred_key == "codes" and "code_ids" in data:
            return data["code_ids"], "code_ids"
        raise KeyError(
            f"Key '{preferred_key}' not found in {npz_path}. Available keys: {list(data.keys())}"
        )


def extract_features(x, feature_mode):
    x = np.asarray(x, dtype=np.float32)

    if feature_mode == "flat":
        return x.reshape(x.shape[0], -1)

    if feature_mode == "pooled":
        if x.ndim == 4:
            return x.mean(axis=(1, 3))
        if x.ndim == 3:
            return x.mean(axis=1)
        return x.reshape(x.shape[0], -1)

    if feature_mode == "pooled_stats":
        if x.ndim == 4:
            mean = x.mean(axis=(1, 3))
            std = x.std(axis=(1, 3))
            maxv = x.max(axis=(1, 3))
            minv = x.min(axis=(1, 3))
            return np.concatenate([mean, std, maxv, minv], axis=1)
        if x.ndim == 3:
            mean = x.mean(axis=1)
            std = x.std(axis=1)
            maxv = x.max(axis=1)
            minv = x.min(axis=1)
            return np.concatenate([mean, std, maxv, minv], axis=1)
        flat = x.reshape(x.shape[0], -1)
        return np.concatenate([flat, np.square(flat), np.abs(flat)], axis=1)

    if feature_mode == "pooled_rich":
        if x.ndim == 4:
            mean = x.mean(axis=(1, 3))
            std = x.std(axis=(1, 3))
            maxv = x.max(axis=(1, 3))
            minv = x.min(axis=(1, 3))
            q25 = np.quantile(x, 0.25, axis=(1, 3))
            q75 = np.quantile(x, 0.75, axis=(1, 3))
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
        f"Unknown feature_mode '{feature_mode}'. Use one of: flat, pooled, pooled_stats, pooled_rich"
    )


def evaluate_binary(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(np.int64)
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    return float(acc), float(auc), y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="vqvae_classifier_config.yaml", type=str)
    parser.add_argument("--split", default="val", type=str, choices=["train", "val", "test"])
    parser.add_argument("--model-file", default=None, type=str)
    parser.add_argument("--code-path", default=None, type=str)
    parser.add_argument("--label-path", default=None, type=str)
    parser.add_argument("--threshold", default=0.5, type=float)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    xgb_cfg = cfg.get("code_xgboost", cfg.get("code_classifier", {}))

    base_path = xgb_cfg.get("base_path", "IQ_labeled_data/June_downsampling_10ms")
    code_suffix = xgb_cfg.get("code_suffix", "revin_x_quantized_codes.npz")
    label_suffix = xgb_cfg.get("label_suffix", "notrevin_x_binary_labels.npz")
    code_key = xgb_cfg.get("code_key", "codes")
    label_key = xgb_cfg.get("label_key", "binary_classes")
    target_prn = int(xgb_cfg.get("target_prn", 29))
    save_path = xgb_cfg.get("save_path", "saved_models/code_classifier")

    model_file = args.model_file or os.path.join(save_path, "code_xgboost.pkl")
    code_path = args.code_path or os.path.join(base_path, f"{args.split}_{code_suffix}")
    label_path = args.label_path or os.path.join(base_path, f"{args.split}_{label_suffix}")

    with open(model_file, "rb") as f:
        model_bundle = pickle.load(f)

    codes, resolved_key = load_code_array(code_path, code_key)
    if resolved_key != code_key:
        print(f"[Info] Using fallback code key '{resolved_key}' in {code_path}")

    if isinstance(model_bundle, dict) and model_bundle.get("model_type") == "xgboost_greedy_ensemble":
        members = model_bundle["members"]
    else:
        members = [{"feature_mode": "pooled_stats", "model": model_bundle}]

    feat_cache = {}
    probs = []
    for member in members:
        mode = str(member["feature_mode"])
        if mode not in feat_cache:
            feat_cache[mode] = extract_features(codes, mode)
        prob = member["model"].predict_proba(feat_cache[mode])[:, 1]
        probs.append(prob)

    y_prob = np.mean(np.stack(probs, axis=0), axis=0)
    y_pred = (y_prob >= args.threshold).astype(np.int64)

    print(f"Samples: {len(y_prob)}")
    print(f"Model members: {len(members)}")
    print(f"Feature modes used: {sorted(list(set(str(m['feature_mode']) for m in members)))}")

    if os.path.exists(label_path):
        labels_raw = load_npz_array(label_path, label_key)[:, target_prn - 1]
        unique = sorted(list(set(labels_raw.tolist())))
        label_map = {int(v): i for i, v in enumerate(unique)}
        y_true = np.array([label_map[int(v)] for v in labels_raw], dtype=np.int64)
        acc, auc, y_pred = evaluate_binary(y_true, y_prob, threshold=args.threshold)
        print(f"Eval ({args.split}) | acc={acc:.4f} auc={auc:.4f}")
    else:
        print(f"[Warn] Label file not found: {label_path}. Skipping metrics.")

    os.makedirs(save_path, exist_ok=True)
    out_file = os.path.join(save_path, f"xgboost_predictions_{args.split}.npz")
    np.savez(
        out_file,
        probs=y_prob.astype(np.float32),
        preds=y_pred.astype(np.int64),
    )
    print(f"Saved predictions to: {out_file}")


if __name__ == "__main__":
    main()
