import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate quantized codes for 10ms IQ sequences using a 1ms VQ-VAE model."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="IQ_labeled_data/June_downsampling_10ms",
        help="Directory containing *_x.npy files.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="saved_models/model_down_160_if_compasation_low_pass/checkpoints/final_model.pth",
        help="Path to trained VQ-VAE model checkpoint (.pth).",
    )
    parser.add_argument(
        "--input-files",
        nargs="+",
        default=["train_revin_x.npy", "val_revin_x.npy", "test_revin_x.npy"],
        help="Input .npy files (relative to --data-dir).",
    )
    parser.add_argument(
        "--piece-length",
        type=int,
        default=1024,
        help="Length (time axis) for each 1ms piece.",
    )
    parser.add_argument(
        "--num-pieces",
        type=int,
        default=10,
        help="Number of 1ms pieces per sample.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for model inference over 1ms pieces.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Inference device.",
    )
    return parser.parse_args()


def chunk_into_pieces(x: np.ndarray, num_pieces: int, piece_length: int) -> np.ndarray:
    # x: [N, C, T]
    if x.ndim != 3:
        raise ValueError(f"Expected input with 3 dims [N,C,T], got shape {x.shape}")

    n_samples, _, total_t = x.shape
    expected_t = num_pieces * piece_length
    if total_t != expected_t:
        raise ValueError(
            f"Time length mismatch: got T={total_t}, expected {num_pieces}*{piece_length}={expected_t}"
        )

    # [N, C, num_pieces, piece_length] -> [N, num_pieces, C, piece_length]
    pieces = x.reshape(n_samples, x.shape[1], num_pieces, piece_length).transpose(0, 2, 1, 3)
    return pieces


def infer_codes(model, pieces: np.ndarray, batch_size: int, device: str):
    # pieces: [N, P, C, T_piece]
    n_samples, num_pieces, channels, t_piece = pieces.shape
    flat = pieces.reshape(n_samples * num_pieces, channels, t_piece)

    all_codes = []
    all_code_ids = []
    with torch.no_grad():
        for start in tqdm(range(0, flat.shape[0], batch_size), desc="Infer 1ms pieces"):
            end = min(start + batch_size, flat.shape[0])
            batch = torch.from_numpy(flat[start:end]).to(device=device, dtype=torch.float32)
            codes, _, code_ids, _ = model.revintime2codes(batch)
            # codes: [B, 1, code_dim, compressed_t]
            codes = codes.squeeze(1).detach().cpu().numpy().astype(np.float32)
            # code_ids: [B, 1, compressed_t]
            code_ids = code_ids.squeeze(1).detach().cpu().numpy().astype(np.int32)
            all_codes.append(codes)
            all_code_ids.append(code_ids)

    flat_codes = np.concatenate(all_codes, axis=0)
    flat_code_ids = np.concatenate(all_code_ids, axis=0)
    # [N*P, code_dim, compressed_t] -> [N, P, code_dim, compressed_t]
    codes = flat_codes.reshape(n_samples, num_pieces, flat_codes.shape[-2], flat_codes.shape[-1])
    # [N*P, compressed_t] -> [N, P, compressed_t]
    code_ids = flat_code_ids.reshape(n_samples, num_pieces, flat_code_ids.shape[-1])
    return codes, code_ids


def process_file(model, data_dir: Path, file_name: str, num_pieces: int, piece_length: int, batch_size: int, device: str):
    in_path = data_dir / file_name
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    x = np.load(in_path, allow_pickle=True).astype(np.float32)
    pieces = chunk_into_pieces(x, num_pieces=num_pieces, piece_length=piece_length)
    codes, code_ids = infer_codes(model, pieces, batch_size=batch_size, device=device)

    out_name = file_name.replace(".npy", "_quantized_codes.npz")
    out_path = data_dir / out_name
    np.savez_compressed(
        out_path,
        codes=codes,
        code_ids=code_ids,
        num_pieces=np.int32(num_pieces),
        original_shape=np.array(x.shape, dtype=np.int32),
    )

    print(f"Saved: {out_path}")
    print(f"  input shape: {x.shape}")
    print(f"  codes shape: {codes.shape}")
    print(f"  code_ids shape: {code_ids.shape}")


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    model_path = Path(args.model_path)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    model.to(device)

    print(f"Device: {device}")
    print(f"Model: {model_path}")
    print(f"Data dir: {data_dir}")

    for file_name in args.input_files:
        print(f"\nProcessing {file_name} ...")
        process_file(
            model=model,
            data_dir=data_dir,
            file_name=file_name,
            num_pieces=args.num_pieces,
            piece_length=args.piece_length,
            batch_size=args.batch_size,
            device=str(device),
        )


if __name__ == "__main__":
    main()
