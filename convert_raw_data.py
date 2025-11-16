import numpy as np
import os
import pdb
import random
import torch

from omegaconf import OmegaConf
from data_provider.data_factory import data_provider
from layers.RevIN import RevIN


class ExtractData:
    def __init__(self, args):
        self.args = args
        self.device = 'cuda:' + str(self.args.gpu)
        self.revin_layer_x = RevIN(num_features=self.args.enc_in, affine=False, subtract_last=False)

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def one_loop(self, loader):
        x_original = []
        x_in_revin_space = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
            x_original.append(np.array(batch_x))
            batch_x = batch_x.float().to(self.device)

            # data going into revin should have dim:[bs x seq_len x nvars]
            x_in_revin_space.append(np.array(self.revin_layer_x(batch_x, "norm").detach().cpu()))

        x_original_arr = np.concatenate(x_original, axis=0)
        x_in_revin_space_arr = np.concatenate(x_in_revin_space, axis=0)

        print(x_in_revin_space_arr.shape, x_original_arr.shape)
        return x_in_revin_space_arr, x_original_arr

    def extract_data(self):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        print('got loaders starting revin')

        # These have dimension [bs, ntime, nvars]
        x_train_in_revin_space_arr, x_train_original_arr = self.one_loop(train_loader)
        print('starting val')
        x_val_in_revin_space_arr, x_val_original_arr = self.one_loop(vali_loader)
        print('starting test')
        x_test_in_revin_space_arr, x_test_original_arr = self.one_loop(test_loader)

        print('Flattening Sensors Out')
        if self.args.seq_len != 96 and self.args.pred_len != 0 :
            pdb.set_trace()
        else:
            # These have dimension [bs, ntime, channels, 2] --> reshape to [bs, ntime, channels*2]
            x_train_arr = x_train_in_revin_space_arr.reshape((x_train_in_revin_space_arr.shape[0],
                                                              x_train_in_revin_space_arr.shape[1],
                                                              -1))
            x_val_arr = x_val_in_revin_space_arr.reshape((x_val_in_revin_space_arr.shape[0],
                                                          x_val_in_revin_space_arr.shape[1],
                                                            -1))
            x_test_arr = x_test_in_revin_space_arr.reshape((x_test_in_revin_space_arr.shape[0],
                                                            x_test_in_revin_space_arr.shape[1],
                                                            -1))

            orig_x_train_arr = x_train_original_arr.reshape((x_train_original_arr.shape[0],
                                                             x_train_original_arr.shape[1],
                                                             -1))
            orig_x_val_arr = x_val_original_arr.reshape((x_val_original_arr.shape[0],
                                                         x_val_original_arr.shape[1],
                                                         -1))
            orig_x_test_arr = x_test_original_arr.reshape((x_test_original_arr.shape[0],
                                                           x_test_original_arr.shape[1],
                                                           -1))
            
            x_train_arr = np.swapaxes(x_train_arr, 1, 2)
            x_val_arr = np.swapaxes(x_val_arr, 1, 2)
            x_test_arr = np.swapaxes(x_test_arr, 1, 2)  
            orig_x_train_arr = np.swapaxes(orig_x_train_arr, 1, 2)
            orig_x_val_arr = np.swapaxes(orig_x_val_arr, 1, 2)
            orig_x_test_arr = np.swapaxes(orig_x_test_arr, 1, 2)

            print(x_train_arr.shape, x_val_arr.shape, x_test_arr.shape)
            print(orig_x_train_arr.shape, orig_x_val_arr.shape, orig_x_test_arr.shape)

        if not os.path.exists(self.args.save_path):
            os.makedirs(self.args.save_path)

        np.save(self.args.save_path + '/train_revin_x.npy', x_train_arr)
        np.save(self.args.save_path + '/val_revin_x.npy', x_val_arr)
        np.save(self.args.save_path + '/test_revin_x.npy', x_test_arr)

        np.save(self.args.save_path + '/train_notrevin_x.npy', orig_x_train_arr)
        np.save(self.args.save_path + '/val_notrevin_x.npy', orig_x_val_arr)
        np.save(self.args.save_path + '/test_notrevin_x.npy', orig_x_test_arr)


if __name__ == '__main__':
    config = OmegaConf.load('dataset_config.yaml')

    # random seed
    fix_seed = config.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    config.use_gpu = True if torch.cuda.is_available() and config.use_gpu else False

    if config.use_gpu and config.use_multi_gpu:
        config.devices = config.devices.replace(' ', '')
        device_ids = config.devices.split(',')
        config.device_ids = [int(id_) for id_ in device_ids]
        config.gpu = config.device_ids[0]


    print('Args in experiment:')
    print(config)

    Exp = ExtractData
    exp = Exp(config)  # set experiments
    exp.extract_data()
