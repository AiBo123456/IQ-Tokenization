import argparse
import numpy as np
import os
import pdb
from omegaconf import OmegaConf
import torch
import torch.nn as nn


def one_loop(loader, vqvae_model, device, args):
    mse = []
    mae = []
    acc = []

    for i, batch_x in enumerate(loader):
        batch_x = batch_x.float().to(device)
        # batch_x = torch.unsqueeze(batch_x, dim=1)  # expects time to be dim [bs x nvars x time]

        # random mask
        B, N, T = batch_x.shape
        mask = torch.rand((B, N, T)).to(device)
        mask[mask <= args.mask_ratio] = 0  # masked
        mask[mask > args.mask_ratio] = 1  # remained
        inp = batch_x.masked_fill(mask == 0, 0)

        x_codes, x_code_ids, codebook = revintime2codes(inp, args.compression_factor, vqvae_model.encoder, vqvae_model.vq)
        # expects code to be dim [bs x nvars x compressed_time]
        x_predictions_revin_space = codes2timerevin(x_code_ids, codebook, args.compression_factor, args.data_channels, vqvae_model.decoder)

        batch_x_masky = batch_x[mask == 0]
        pred_x_masky = np.swapaxes(x_predictions_revin_space, 1, 2)[mask == 0]

        mse.append(nn.functional.mse_loss(batch_x_masky, pred_x_masky).item())
        mae.append(nn.functional.l1_loss(batch_x_masky, pred_x_masky).item())
        acc.append(torch.mean((torch.abs(batch_x_masky - pred_x_masky) < 0.1).float()).item())

    print('MSE:', np.mean(mse))
    print('MAE:', np.mean(mae))
    print('Accuracy:', np.mean(acc))

def revintime2codes(revin_data, compression_factor, vqvae_encoder, vqvae_quantizer):
    '''
    Args:
        revin_data: [bs x nvars x pred_len or seq_len]
        compression_factor: int
        vqvae_model: trained vqvae model
        use_grad: bool, if True use gradient, if False don't use gradients

    Returns:
        codes: [bs, nvars, code_dim, compressed_time]
        code_ids: [bs, nvars, compressed_time]
        embedding_weight: [num_code_words, code_dim]

    Helpful VQVAE Comments:
        # Into the vqvae encoder: batch.shape: [bs x seq_len] i.e. torch.Size([256, 12])
        # into the quantizer: z.shape: [bs x code_dim x (seq_len/compresion_factor)] i.e. torch.Size([256, 64, 3])
        # into the vqvae decoder: quantized.shape: [bs x code_dim x (seq_len/compresion_factor)] i.e. torch.Size([256, 64, 3])
        # out of the vqvae decoder: data_recon.shape: [bs x seq_len] i.e. torch.Size([256, 12])
        # this is if your compression factor=4
    '''

    bs = revin_data.shape[0]
    nvar = revin_data.shape[1]
    T = revin_data.shape[2]  # this can be either the prediction length or the sequence length
    compressed_time = int(T / compression_factor)  # this can be the compressed time of either the prediction length or the sequence length

    with torch.no_grad():
        # flat_revin = revin_data.reshape(-1, T)  # flat_y: [bs, nvars, T]
        latent = vqvae_encoder(revin_data.to(torch.float), compression_factor)  # latent_y: [bs, code_dim, compressed_time]
        vq_loss, quantized, perplexity, embedding_weight, encoding_indices, encodings = vqvae_quantizer(latent)  # quantized: [bs, code_dim, compressed_time]
        code_dim = quantized.shape[-2]
        codes = quantized.reshape(bs, 1, code_dim,
                                  compressed_time)  # codes: [bs, nvars, code_dim, compressed_time]
        code_ids = encoding_indices.view(bs, 1, compressed_time)  # code_ids: [bs, nvars, compressed_time]

    return codes, code_ids, embedding_weight


def codes2timerevin(code_ids, codebook, compression_factor, data_channels, vqvae_decoder):
    '''
    Args:
        code_ids: [bs x nvars x compressed_pred_len]
        codebook: [num_code_words, code_dim]
        compression_factor: int
        vqvae_model: trained vqvae model
        use_grad: bool, if True use gradient, if False don't use gradients
        x_or_y: if 'x' use revin_denorm_x if 'y' use revin_denorm_y
    Returns:
        predictions_revin_space: [bs x original_time_len x nvars]
        predictions_original_space: [bs x original_time_len x nvars]
    '''
    bs = code_ids.shape[0]
    nvars = code_ids.shape[1]
    compressed_len = code_ids.shape[2]
    num_code_words = codebook.shape[0]
    code_dim = codebook.shape[1]
    device = code_ids.device
    input_shape = (bs * nvars, compressed_len, code_dim)

    with torch.no_grad():
        # scatter the label with the codebook
        one_hot_encodings = torch.zeros(int(bs * nvars * compressed_len), num_code_words, device=device)  # one_hot_encodings: [bs x nvars x compressed_pred_len, num_codes]
        one_hot_encodings.scatter_(1, code_ids.reshape(-1, 1).to(device),1)  # one_hot_encodings: [bs x nvars x compressed_pred_len, num_codes]
        quantized = torch.matmul(one_hot_encodings, torch.tensor(codebook)).view(input_shape)  # quantized: [bs * nvars, compressed_pred_len, code_dim]
        quantized_swaped = torch.swapaxes(quantized, 1,2)  # quantized_swaped: [bs * nvars, code_dim, compressed_pred_len]
        prediction_recon = vqvae_decoder(quantized_swaped.to(device), compression_factor)  # prediction_recon: [bs * nvars, pred_len]
        prediction_recon_reshaped = prediction_recon.reshape(bs, data_channels, prediction_recon.shape[-1])  # prediction_recon_reshaped: [bs x nvars x pred_len]
        predictions_revin_space = torch.swapaxes(prediction_recon_reshaped, 1,2)  # prediction_recon_nvars_last: [bs x pred_len x nvars]

    return predictions_revin_space


def create_NONrevin_dataloaders(batchsize=100, dataset="dummy", base_path='dummy'):

    test_data = np.load(os.path.join(base_path, "test_notrevin_x.npy"), allow_pickle=True)

    test_dataloader = torch.utils.data.DataLoader(test_data,
                                                batch_size=batchsize,
                                                shuffle=False,
                                                num_workers=10,
                                                drop_last=False)

    return test_dataloader


def main(args):
    device = 'cuda:' + str(args.gpu)
    vqvae_model = torch.load(args.trained_vqvae_model_path, weights_only=False)
    vqvae_model.to(device)
    vqvae_model.eval()

    test_loader = create_NONrevin_dataloaders(batchsize=8192, dataset='IQ_data', base_path=args.base_path)

    print('TEST')
    one_loop(test_loader, vqvae_model, device, args)
    print('-------------')


if __name__ == '__main__':
    args = OmegaConf.load('imputation_config.yaml')

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)
    main(args)
