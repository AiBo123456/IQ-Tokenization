import argparse
import json
import os
import random
import sys
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ofdm_injector import generate_ofdm_signal, inject_anomaly


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_split_path(base_path: str, split: str, revin: bool) -> str:
    suffix = "revin_x.npy" if revin else "notrevin_x.npy"
    path = os.path.join(base_path, f"{split}_{suffix}")
    if not os.path.exists(path) and not revin:
        fallback = os.path.join(base_path, f"{split}_not_revin_x.npy")
        if os.path.exists(fallback):
            return fallback
    return path


def load_windows(path: str) -> np.ndarray:
    windows = np.load(path, allow_pickle=True).astype(np.float32)
    if windows.ndim != 3:
        raise ValueError(f"Expected [N, C, T] input but got shape {windows.shape} from: {path}")
    if windows.shape[1] != 2:
        raise ValueError(f"Expected IQ channels with shape [N, 2, T], got {windows.shape} from: {path}")
    return windows


def inject_ofdm_windows(
    clean_windows: np.ndarray,
    fs: float,
    num_subcarriers: int,
    num_symbols: int,
    cp_length: int,
    freq_offset_hz: float,
    snr_db: float,
) -> tuple[np.ndarray, np.ndarray]:
    n, _, t = clean_windows.shape
    anomaly_len = (num_subcarriers + cp_length) * num_symbols
    if anomaly_len > t:
        raise ValueError(
            "OFDM anomaly is longer than one window. "
            f"anomaly_len={anomaly_len}, window_len={t}. "
            "Reduce num_symbols or num_subcarriers."
        )

    corrupted = clean_windows.copy()
    start_indices = np.empty(n, dtype=np.int32)

    for idx in range(n):
        anomaly_waveform = generate_ofdm_signal(
            num_subcarriers=num_subcarriers,
            num_symbols=num_symbols,
            cp_length=cp_length,
        )
        iq_complex = corrupted[idx, 0] + 1j * corrupted[idx, 1]
        injected_complex, start_idx = inject_anomaly(
            base_signal=iq_complex,
            fs=fs,
            ofdm_signal=anomaly_waveform,
            freq_offset_hz=freq_offset_hz,
            snr_db=snr_db,
        )
        corrupted[idx, 0] = injected_complex.real.astype(np.float32)
        corrupted[idx, 1] = injected_complex.imag.astype(np.float32)
        start_indices[idx] = start_idx

    return corrupted, start_indices


def save_split(
    output_dir: str,
    split: str,
    revin: bool,
    windows: np.ndarray,
    start_indices: np.ndarray,
) -> dict[str, Any]:
    suffix = "revin_x_anomaly.npy" if revin else "notrevin_x_anomaly.npy"
    array_path = os.path.join(output_dir, f"{split}_{suffix}")
    meta_path = os.path.join(output_dir, f"{split}_{'revin' if revin else 'notrevin'}_ofdm_meta.npz")

    np.save(array_path, windows.astype(np.float32))
    np.savez(
        meta_path,
        injection_start_indices=start_indices.astype(np.int32),
    )

    return {
        "split": split,
        "revin": revin,
        "array_path": array_path,
        "meta_path": meta_path,
        "shape": list(windows.shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate OFDM anomaly IQ windows with the same split structure as June_downsampling."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="anomaly_detection/data/June_downsampling",
        help="Folder containing clean split files such as train_notrevin_x.npy.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="anomaly_detection/data/June_downsampling",
        help="Folder where anomaly split files will be written.",
    )
    parser.add_argument(
        "--revin",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate *_revin_x.npy files. Use --no-revin for *_notrevin_x.npy files.",
    )
    parser.add_argument("--fs", type=float, default=1_000_000.0)
    parser.add_argument("--num-subcarriers", type=int, default=64)
    parser.add_argument("--num-symbols", type=int, default=12)
    parser.add_argument("--cp-length", type=int, default=16)
    parser.add_argument("--freq-offset-hz", type=float, default=100_000.0)
    parser.add_argument("--snr-db", type=float, default=-10.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=0,
        help="Optionally limit each split for a quick smoke run.",
    )
    parser.add_argument(
        "--save-summary",
        type=str,
        default="",
        help="Optional JSON path for generation summary. Defaults to <output-dir>/generation_summary.json.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    # os.makedirs(args.output_dir, exist_ok=True)

    split_summaries = []
    for split in ("train", "val", "test"):
        input_path = resolve_split_path(args.input_dir, split, args.revin)
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Missing split file: {input_path}")

        clean_windows = load_windows(input_path)
        if args.max_samples_per_split and args.max_samples_per_split > 0:
            clean_windows = clean_windows[:args.max_samples_per_split]
        anomaly_windows, start_indices = inject_ofdm_windows(
            clean_windows=clean_windows,
            fs=args.fs,
            num_subcarriers=args.num_subcarriers,
            num_symbols=args.num_symbols,
            cp_length=args.cp_length,
            freq_offset_hz=args.freq_offset_hz,
            snr_db=args.snr_db,
        )
        split_summary = save_split(
            output_dir=args.output_dir,
            split=split,
            revin=args.revin,
            windows=anomaly_windows,
            start_indices=start_indices,
        )
        split_summary["input_path"] = input_path
        split_summary["num_samples"] = int(clean_windows.shape[0])
        split_summary["window_length"] = int(clean_windows.shape[-1])
        split_summaries.append(split_summary)

        print(
            f"Saved {split} split to {split_summary['array_path']} "
            f"with shape {tuple(anomaly_windows.shape)}"
        )

    summary = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "revin": args.revin,
        "seed": args.seed,
        "ofdm_config": {
            "fs": args.fs,
            "num_subcarriers": args.num_subcarriers,
            "num_symbols": args.num_symbols,
            "cp_length": args.cp_length,
            "freq_offset_hz": args.freq_offset_hz,
            "snr_db": args.snr_db,
        },
        "splits": split_summaries,
    }

    summary_path = args.save_summary or os.path.join(args.output_dir, "generation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved generation summary to {summary_path}")


if __name__ == "__main__":
    main()
