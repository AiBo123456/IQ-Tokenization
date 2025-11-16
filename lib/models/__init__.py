from .vqvae import vqvae
from .vqvae_shuflle_net import vqvae_shuffle_net

model_dict = {
    'vqvae': vqvae,
    'vqvae_shuffle_net': vqvae_shuffle_net
}

def get_model_class(model_name):
    if model_name in model_dict:
        return model_dict[model_name]
    else:
        raise NotImplementedError
