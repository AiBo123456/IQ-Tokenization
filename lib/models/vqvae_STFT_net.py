import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.core import BaseModel


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv1d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim, compression_factor):
        super(Encoder, self).__init__()
        if compression_factor == 4:
            self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_hiddens // 2,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_2 = nn.Conv1d(in_channels=num_hiddens // 2,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_3 = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=3,
                                     stride=1, padding=1)
            self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens)

            self._pre_vq_conv = nn.Conv1d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)

        elif compression_factor == 8:
            self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_hiddens // 2,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_2 = nn.Conv1d(in_channels=num_hiddens // 2,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_A = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_3 = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=3,
                                     stride=1, padding=1)
            self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens)

            self._pre_vq_conv = nn.Conv1d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)

        elif compression_factor == 12:
            self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_hiddens // 2,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_2 = nn.Conv1d(in_channels=num_hiddens // 2,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_3 = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=3, padding=1)
            self._conv_4 = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=3,
                                     stride=1, padding=1)
            self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens)

            self._pre_vq_conv = nn.Conv1d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)

        elif compression_factor == 16:
            self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_hiddens // 2,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_2 = nn.Conv1d(in_channels=num_hiddens // 2,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_A = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_B = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_3 = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=3,
                                     stride=1, padding=1)
            self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens)

            self._pre_vq_conv = nn.Conv1d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)

    def forward(self, inputs, compression_factor):
        
        if compression_factor == 4:
            x = inputs

            x = self._conv_1(x)
            x = F.relu(x)

            x = self._conv_2(x)
            x = F.relu(x)

            x = self._conv_3(x)
            x = self._residual_stack(x)
            x = self._pre_vq_conv(x)

            return x

        elif compression_factor == 8:
            x = inputs

            x = self._conv_1(x)
            x = F.relu(x)

            x = self._conv_2(x)
            x = F.relu(x)

            x = self._conv_A(x)
            x = F.relu(x)

            x = self._conv_3(x)
            x = self._residual_stack(x)
            x = self._pre_vq_conv(x)
            return x

        elif compression_factor == 12:
            x = inputs

            x = self._conv_1(x)
            x = F.relu(x)

            x = self._conv_2(x)
            x = F.relu(x)

            x = self._conv_3(x)
            x = F.relu(x)

            x = self._conv_4(x)
            x = self._residual_stack(x)
            x = self._pre_vq_conv(x)
            return x

        elif compression_factor == 16:
            # x = inputs.view([inputs.shape[0], 1, inputs.shape[-1]])
            x = inputs

            x = self._conv_1(x)
            x = F.relu(x)

            x = self._conv_2(x)
            x = F.relu(x)

            x = self._conv_A(x)
            x = F.relu(x)

            x = self._conv_B(x)
            x = F.relu(x)

            x = self._conv_3(x)
            x = self._residual_stack(x)
            x = self._pre_vq_conv(x)
            return x

class Normal_Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim, compression_factor):
        super(Normal_Encoder, self).__init__()
        if compression_factor == 4:
            self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_hiddens // 2,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_2 = nn.Conv1d(in_channels=num_hiddens // 2,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_3 = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=3,
                                     stride=1, padding=1)
            self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens)

            self._pre_vq_conv = nn.Conv1d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)

        elif compression_factor == 8:
            self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_hiddens // 2,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_2 = nn.Conv1d(in_channels=num_hiddens // 2,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_A = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_3 = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=3,
                                     stride=1, padding=1)
            self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens)

            self._pre_vq_conv = nn.Conv1d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)

        elif compression_factor == 12:
            self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_hiddens // 2,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_2 = nn.Conv1d(in_channels=num_hiddens // 2,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_3 = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=3, padding=1)
            self._conv_4 = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=3,
                                     stride=1, padding=1)
            self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens)

            self._pre_vq_conv = nn.Conv1d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)

        elif compression_factor == 16:
            self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_hiddens // 2,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_2 = nn.Conv1d(in_channels=num_hiddens // 2,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_A = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_B = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_3 = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=3,
                                     stride=1, padding=1)
            self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens)

            self._pre_vq_conv = nn.Conv1d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)

    def forward(self, inputs, compression_factor):
        
        if compression_factor == 4:
            x = inputs

            x = self._conv_1(x)
            x = F.relu(x)

            x = self._conv_2(x)
            x = F.relu(x)

            x = self._conv_3(x)
            x = self._residual_stack(x)
            x = self._pre_vq_conv(x)

            B, C, T = x.shape
            x = F.normalize(x.reshape(B, -1), dim=-1)
            x = x.reshape(B, C, T)
            return x

        elif compression_factor == 8:
            x = inputs

            x = self._conv_1(x)
            x = F.relu(x)

            x = self._conv_2(x)
            x = F.relu(x)

            x = self._conv_A(x)
            x = F.relu(x)

            x = self._conv_3(x)
            x = self._residual_stack(x)
            x = self._pre_vq_conv(x)

            B, C, T = x.shape
            x = F.normalize(x.reshape(B, -1), dim=-1)
            x = x.reshape(B, C, T)
            return x

        elif compression_factor == 12:
            x = inputs

            x = self._conv_1(x)
            x = F.relu(x)

            x = self._conv_2(x)
            x = F.relu(x)

            x = self._conv_3(x)
            x = F.relu(x)

            x = self._conv_4(x)
            x = self._residual_stack(x)
            x = self._pre_vq_conv(x)

            B, C, T = x.shape
            x = F.normalize(x.reshape(B, -1), dim=-1)
            x = x.reshape(B, C, T)
            return x

        elif compression_factor == 16:
            # x = inputs.view([inputs.shape[0], 1, inputs.shape[-1]])
            x = inputs

            x = self._conv_1(x)
            x = F.relu(x)

            x = self._conv_2(x)
            x = F.relu(x)

            x = self._conv_A(x)
            x = F.relu(x)

            x = self._conv_B(x)
            x = F.relu(x)

            x = self._conv_3(x)
            x = self._residual_stack(x)
            x = self._pre_vq_conv(x)

            B, C, T = x.shape
            x = F.normalize(x.reshape(B, -1), dim=-1)
            x = x.reshape(B, C, T)
            return x


class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, compression_factor, output_channels=1):
        super(Decoder, self).__init__()
        if compression_factor == 4:
            self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_hiddens,
                                     kernel_size=3,
                                     stride=1, padding=1)

            self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens)

            self._conv_trans_1 = nn.ConvTranspose1d(in_channels=num_hiddens,
                                                    out_channels=max(num_hiddens // 2, output_channels),
                                                    kernel_size=4,
                                                    stride=2, padding=1)

            self._conv_trans_2 = nn.ConvTranspose1d(in_channels=max(num_hiddens // 2, output_channels),
                                                    out_channels=output_channels,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

        elif compression_factor == 8:
            self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_hiddens,
                                     kernel_size=3,
                                     stride=1, padding=1)

            self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens)

            self._conv_trans_A = nn.ConvTranspose1d(in_channels=num_hiddens,
                                                    out_channels=num_hiddens,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

            self._conv_trans_1 = nn.ConvTranspose1d(in_channels=num_hiddens,
                                                    out_channels=max(num_hiddens // 2, output_channels),
                                                    kernel_size=4,
                                                    stride=2, padding=1)

            self._conv_trans_2 = nn.ConvTranspose1d(in_channels=max(num_hiddens // 2, output_channels),
                                                    out_channels=1,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

        elif compression_factor == 12:
            self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_hiddens,
                                     kernel_size=3,
                                     stride=1, padding=1)

            self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens)

            # To get the correct shape back the kernel size has to be 5 not 4
            self._conv_trans_2 = nn.ConvTranspose1d(in_channels=num_hiddens,
                                                    out_channels=num_hiddens,
                                                    kernel_size=5,
                                                    stride=3, padding=1)

            self._conv_trans_3 = nn.ConvTranspose1d(in_channels=num_hiddens,
                                                    out_channels=max(num_hiddens // 2, output_channels),
                                                    kernel_size=4,
                                                    stride=2, padding=1)

            self._conv_trans_4 = nn.ConvTranspose1d(in_channels=max(num_hiddens // 2, output_channels),
                                                    out_channels=output_channels,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

        elif compression_factor == 16:
            self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_hiddens,
                                     kernel_size=3,
                                     stride=1, padding=1)

            self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens)

            self._conv_trans_A = nn.ConvTranspose1d(in_channels=num_hiddens,
                                                    out_channels=num_hiddens,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

            self._conv_trans_B = nn.ConvTranspose1d(in_channels=num_hiddens,
                                                    out_channels=num_hiddens,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

            self._conv_trans_1 = nn.ConvTranspose1d(in_channels=num_hiddens,
                                                    out_channels=max(num_hiddens // 2, output_channels),
                                                    kernel_size=4,
                                                    stride=2, padding=1)

            self._conv_trans_2 = nn.ConvTranspose1d(in_channels=max(num_hiddens // 2, output_channels),
                                                    out_channels=output_channels,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

    def forward(self, inputs, compression_factor):
        if compression_factor == 4:
            x = self._conv_1(inputs)

            x = self._residual_stack(x)

            x = self._conv_trans_1(x)
            x = F.relu(x)

            x = self._conv_trans_2(x)

            return x

        elif compression_factor == 8:
            x = self._conv_1(inputs)

            x = self._residual_stack(x)

            x = self._conv_trans_A(x)
            x = F.relu(x)

            x = self._conv_trans_1(x)
            x = F.relu(x)

            x = self._conv_trans_2(x)

            return x

        elif compression_factor == 12:
            x = self._conv_1(inputs)
            x = self._residual_stack(x)

            x = self._conv_trans_2(x)
            x = F.relu(x)

            x = self._conv_trans_3(x)
            x = F.relu(x)

            x = self._conv_trans_4(x)

            return x

        elif compression_factor == 16:
            x = self._conv_1(inputs)

            x = self._residual_stack(x)

            x = self._conv_trans_A(x)
            x = F.relu(x)

            x = self._conv_trans_B(x)
            x = F.relu(x)

            x = self._conv_trans_1(x)
            x = F.relu(x)

            x = self._conv_trans_2(x)

            return x


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True) + torch.sum(self._embedding.weight ** 2, dim=1) - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return loss, quantized.permute(0, 2, 1).contiguous(), perplexity, self._embedding.weight, encoding_indices, encodings


class vqvae_L1_STFT_net(BaseModel):
    def __init__(self, vqvae_config):
        super().__init__()
        num_hiddens = vqvae_config['block_hidden_size']
        num_residual_layers = vqvae_config['num_residual_layers']
        num_residual_hiddens = vqvae_config['res_hidden_size']
        embedding_dim = vqvae_config['embedding_dim']
        num_embeddings = vqvae_config['num_embeddings']
        commitment_cost = vqvae_config['commitment_cost']
        data_channels = vqvae_config['data_channels']
        self.compression_factor = vqvae_config['compression_factor']
        self.infoNCE_factor = vqvae_config['infoNCE_factor']
        self.stft_loss_factor = vqvae_config.get('stft_loss_factor', 1.0)
        self.stft_n_fft = vqvae_config.get('stft_n_fft', 64)
        self.stft_hop_length = vqvae_config.get('stft_hop_length', self.stft_n_fft // 4)
        self.stft_win_length = vqvae_config.get('stft_win_length', self.stft_n_fft)

        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.encoder = Encoder(data_channels, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim, self.compression_factor)
        self.decoder = Decoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens, self.compression_factor, output_channels=data_channels)

    def _log_cosh_loss(self, prediction, target):
        diff = prediction - target
        # Stable equivalent of log(cosh(diff)).
        return (diff + F.softplus(-2.0 * diff) - math.log(2.0)).mean()

    def _stft_recon_loss(self, prediction, target):
        if self.stft_loss_factor <= 0:
            return prediction.new_zeros(())

        _, num_channels, signal_length = prediction.shape
        n_fft = min(self.stft_n_fft, signal_length)
        if n_fft < 2:
            return prediction.new_zeros(())

        win_length = min(self.stft_win_length, n_fft)
        hop_length = max(1, min(self.stft_hop_length, win_length))
        window = torch.hann_window(win_length, device=prediction.device, dtype=prediction.dtype)

        if num_channels == 2:
            # Treat the IQ pair as a single complex waveform: I + jQ.
            prediction_flat = torch.complex(prediction[:, 0, :], prediction[:, 1, :])
            target_flat = torch.complex(target[:, 0, :], target[:, 1, :])
        else:
            prediction_flat = prediction.reshape(-1, signal_length)
            target_flat = target.reshape(-1, signal_length)

        prediction_stft = torch.stft(
            prediction_flat,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
        )
        target_stft = torch.stft(
            target_flat,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
        )
        # return self._log_cosh_loss(prediction_stft.abs(), target_stft.abs())
        return F.mse_loss(prediction_stft.abs(), target_stft.abs())

    def _select_recon_pair(self, prediction, target, reconstruct_only_first=False):
        if not reconstruct_only_first:
            return prediction, target

        half_batch = target.shape[0] // 2
        # if target.shape[0] < 2 or target.shape[0] % 2 != 0:
        #     raise ValueError(
        #         "reconstruct_only_first=True expects an even batch size so the left "
        #         "half can be used as the reconstruction target for both halves."
        #     )

        first_half_target = target[:half_batch]
        repeats = prediction.shape[0] // half_batch
        # if prediction.shape[0] != half_batch * repeats:
        #     raise ValueError(
        #         "Prediction batch size must be an integer multiple of the left-half "
        #         "target batch size when reconstruct_only_first=True."
        #     )

        repeat_shape = (repeats,) + (1,) * (target.ndim - 1)
        return prediction, first_half_target.repeat(repeat_shape)

    def _compute_recon_error(self, prediction, target, reconstruct_only_first=False, include_stft=False):
        recon_prediction, recon_target = self._select_recon_pair(
            prediction,
            target,
            reconstruct_only_first=reconstruct_only_first,
        )
        # recon_error = self._log_cosh_loss(recon_prediction, recon_target)
        recon_error = F.mse_loss(recon_prediction, recon_target)
        # recon_error = F.l1_loss(recon_prediction, recon_target)
        if include_stft:
            recon_error = recon_error + self.stft_loss_factor * self._stft_recon_loss(
                recon_prediction,
                recon_target,
            )
        return recon_error

    def revintime2codes(self, batch):
        bs = batch.shape[0]
        nvar = batch.shape[1]
        T = batch.shape[2]  # this can be either the prediction length or the sequence length
        compressed_time = int(T / self.compression_factor)  # this can be the compressed time of either the prediction length or the sequence length

        with torch.no_grad():
            # flat_revin = revin_data.reshape(-1, T)  # flat_y: [bs, nvars, T]
            latent = self.encoder(batch.to(torch.float), self.compression_factor)  # latent_y: [bs, code_dim, compressed_time]
            vq_loss, quantized, perplexity, embedding_weight, encoding_indices, encodings = self.vq(latent)  # quantized: [bs, code_dim, compressed_time]
            code_dim = quantized.shape[-2]
            codes = quantized.reshape(bs, 1, code_dim,
                                    compressed_time)  # codes: [bs, nvars, code_dim, compressed_time]
            code_ids = encoding_indices.view(bs, 1, compressed_time)  # code_ids: [bs, nvars, compressed_time]

        return codes, latent, code_ids, embedding_weight

    def triplet_eval(self, batch, batch_masked, optimizer, mode, comet_logger=None, reconstruct_only_first=False):
        if mode == 'train':
            optimizer.zero_grad()

            z = self.encoder(batch_masked, self.compression_factor)
            vq_loss, quantized, perplexity, embedding_weight, encoding_indices, encodings = self.vq(z)
            data_recon = self.decoder(quantized, self.compression_factor)
            
            # infoNCE loss
            B, C, T = z.shape # (800*2, 64, 250)
            z = z.reshape(3, B//3, C, T)
            z_1_flat = z[0].mean(dim=-1)
            z_2_flat = z[1].mean(dim=-1)
            z_3_flat = torch.roll(z[2].mean(dim=-1), shifts=random.randint(1, B//3-1), dims=0)
            triplet_loss = F.triplet_margin_loss(z_1_flat, z_2_flat, z_3_flat, margin=0.3, p=2)

            recon_error = self._compute_recon_error(
                data_recon,
                batch,
                reconstruct_only_first=reconstruct_only_first,
            )
            loss = recon_error + vq_loss + triplet_loss
            loss.backward()
            optimizer.step()

        return loss, vq_loss, recon_error, data_recon, perplexity, embedding_weight, encoding_indices, encodings, triplet_loss

    def contrastive_eval(self, batch, batch_masked, optimizer, mode, comet_logger=None, reconstruct_only_first=False):
        if mode == 'train':
            optimizer.zero_grad()

            z = self.encoder(batch_masked, self.compression_factor)
            vq_loss, quantized, perplexity, embedding_weight, encoding_indices, encodings = self.vq(z)
            data_recon = self.decoder(quantized, self.compression_factor)
            # data_recon = self.decoder(z, self.compression_factor)
            
            # infoNCE loss
            B, C, T = z.shape # (800*2, 64, 250)
            z = z.reshape(2, B//2, C, T)
            z_1_flat = z[0].mean(dim=-1)
            z_2_flat = z[1].mean(dim=-1)
            z_1_flat = F.normalize(z_1_flat, dim=-1)
            z_2_flat = F.normalize(z_2_flat, dim=-1)
            labels = torch.arange(B//2, device=z_1_flat.device)
            loss = F.cross_entropy(z_1_flat @ z_2_flat.t() / 0.07, labels)
            infoNCE_loss = loss * self.infoNCE_factor
            # infoNCE_loss = torch.tensor(0.0, device=z.device)

            recon_error = self._compute_recon_error(
                data_recon,
                batch,
                reconstruct_only_first=reconstruct_only_first,
                include_stft=True,
            )
            loss = recon_error + vq_loss + infoNCE_loss
            # loss = recon_error + infoNCE_loss
            loss.backward()
            optimizer.step()

        if mode == 'val' or mode == 'test':
            with torch.no_grad():
                z = self.encoder(batch_masked, self.compression_factor)
                vq_loss, quantized, perplexity, embedding_weight, encoding_indices, encodings = self.vq(z)

                data_recon = self.decoder(quantized, self.compression_factor)
                # infoNCE loss
                B, C, T = z.shape
                z = z.reshape(2, B//2, C, T)
                z_1_flat = z[0].reshape(B//2, -1)
                z_2_flat = z[1].reshape(B//2, -1)
                z_1_flat = F.normalize(z_1_flat, dim=-1)
                z_2_flat = F.normalize(z_2_flat, dim=-1)
                labels = torch.arange(B//2, device=z_1_flat.device)
                loss1 = F.cross_entropy(z_1_flat @ z_2_flat.t() / 0.07, labels)
                loss2 = F.cross_entropy(z_2_flat @ z_1_flat.t() / 0.07, labels)
                infoNCE_loss = (loss1 + loss2) / 2 * self.infoNCE_factor


                recon_error = self._compute_recon_error(
                    data_recon,
                    batch,
                    reconstruct_only_first=reconstruct_only_first,
                    include_stft=True,
                )
                loss = recon_error + vq_loss + infoNCE_loss

        # turning this off for faster training - uncomment if want to create loss / perplexity curves
        # comet_logger.log_metric(f'{mode}_vqvae_loss_each_batch', loss.item())
        # comet_logger.log_metric(f'{mode}_vqvae_vq_loss_each_batch', vq_loss.item())
        # comet_logger.log_metric(f'{mode}_vqvae_recon_loss_each_batch', recon_error.item())
        # comet_logger.log_metric(f'{mode}_vqvae_perplexity_each_batch', perplexity.item())

        return loss, vq_loss, recon_error, data_recon, perplexity, embedding_weight, encoding_indices, encodings, infoNCE_loss
    
    def shared_eval(self, batch, batch_masked, optimizer, mode, comet_logger=None, reconstruct_only_first=False):
        if mode == 'train':
            optimizer.zero_grad()

            z = self.encoder(batch_masked, self.compression_factor)
            vq_loss, quantized, perplexity, embedding_weight, encoding_indices, encodings = self.vq(z)
            data_recon = self.decoder(quantized, self.compression_factor)

            recon_error = self._compute_recon_error(
                data_recon,
                batch,
                reconstruct_only_first=reconstruct_only_first,
                # include_stft=True,
            )
            loss = recon_error + vq_loss
            loss.backward()
            optimizer.step()

        if mode == 'val' or mode == 'test':
            with torch.no_grad():
                z = self.encoder(batch_masked, self.compression_factor)
                vq_loss, quantized, perplexity, embedding_weight, encoding_indices, encodings = self.vq(z)

                data_recon = self.decoder(quantized, self.compression_factor)
                recon_error = self._compute_recon_error(
                    data_recon,
                    batch,
                    reconstruct_only_first=reconstruct_only_first,
                    include_stft=True,
                )
                loss = recon_error + vq_loss

        # turning this off for faster training - uncomment if want to create loss / perplexity curves
        # comet_logger.log_metric(f'{mode}_vqvae_loss_each_batch', loss.item())
        # comet_logger.log_metric(f'{mode}_vqvae_vq_loss_each_batch', vq_loss.item())
        # comet_logger.log_metric(f'{mode}_vqvae_recon_loss_each_batch', recon_error.item())
        # comet_logger.log_metric(f'{mode}_vqvae_perplexity_each_batch', perplexity.item())

        return loss, vq_loss, recon_error, data_recon, perplexity, embedding_weight, encoding_indices, encodings
