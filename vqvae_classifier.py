import math
import torch
import torch.nn as nn


def _sinusoidal_positional_encoding(seq_len, d_model, device):
    if d_model % 2 != 0:
        raise ValueError("d_model must be even for sinusoidal positional encoding.")
    positions = torch.arange(seq_len, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term)
    return pe


class VQVAEClassifier(nn.Module):
    def __init__(
        self,
        vqvae,
        num_classes,
        mlp_hidden=128,
        dropout=0.1,
        pooling="mean",
        arch="mlp_flat",
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        reinit_model_params=False,
    ):
        super().__init__()
        self.vqvae = vqvae
        self.pooling = pooling
        self.arch = arch

        # Freeze encoder + quantizer
        if not reinit_model_params:
            for p in self.vqvae.encoder.parameters():
                p.requires_grad = False
            for p in self.vqvae.vq.parameters():
                p.requires_grad = False

        feat_dim = self.vqvae.vq._embedding_dim
        self.flatten_codes = False

        if self.arch == "mlp":
            self.classifier = nn.Sequential(
                nn.Linear(feat_dim, mlp_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, num_classes),
            )
            self.input_proj = None
            self.transformer = None
            self.cls_token = None
            self.out_proj = None
        elif self.arch == "mlp_flat":
            self.flatten_codes = True
            self.classifier = nn.Sequential(
                nn.LazyLinear(mlp_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, mlp_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, num_classes),
            )
            self.input_proj = None
            self.transformer = None
            self.cls_token = None
            self.out_proj = None
        elif self.arch == "transformer":
            self.input_proj = nn.Linear(feat_dim, d_model) if feat_dim != d_model else nn.Identity()

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            if self.pooling == "cls":
                self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            else:
                self.cls_token = None

            self.out_proj = nn.Linear(d_model, num_classes)
            self.classifier = None
        else:
            raise ValueError(f"Unsupported arch: {self.arch}")

    def _extract_quantized(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        with torch.no_grad():
            z = self.vqvae.encoder(x, self.vqvae.compression_factor)
            _, quantized, _, _, _, _ = self.vqvae.vq(z)
        return quantized  # [B, C, T]

    def extract_features(self, x):
        quantized = self._extract_quantized(x)

        if self.pooling == "mean":
            feats = quantized.mean(dim=-1)
        elif self.pooling == "max":
            feats = quantized.max(dim=-1).values
        else:
            raise ValueError(f"Unsupported pooling: {self.pooling}")
        return feats

    def _transformer_logits(self, x):
        quantized = self._extract_quantized(x)
        seq = quantized.permute(0, 2, 1)  # [B, T, C]
        seq = self.input_proj(seq)

        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(seq.size(0), -1, -1)
            seq = torch.cat([cls_tokens, seq], dim=1)

        pos = _sinusoidal_positional_encoding(seq.size(1), seq.size(2), seq.device)
        seq = seq + pos.unsqueeze(0)

        encoded = self.transformer(seq)

        if self.pooling == "cls":
            pooled = encoded[:, 0, :]
        elif self.pooling == "mean":
            pooled = encoded.mean(dim=1)
        elif self.pooling == "max":
            pooled = encoded.max(dim=1).values
        else:
            raise ValueError(f"Unsupported pooling: {self.pooling}")

        return self.out_proj(pooled)

    def _mlp_flat_logits(self, x):
        quantized = self._extract_quantized(x)
        flat = quantized.flatten(1)
        return self.classifier(flat)

    def forward(self, x):
        if self.arch == "mlp":
            feats = self.extract_features(x)
            logits = self.classifier(feats)
            return logits
        if self.arch == "mlp_flat":
            return self._mlp_flat_logits(x)
        return self._transformer_logits(x)
