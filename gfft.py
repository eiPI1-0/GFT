from torch import nn

from .utils import Gaussian


def get_model_layers(model, layer_name):
    layers = model
    for i in layer_name.split('.'):
        layers = getattr(layers, i)
    return layers


class ModelLayerFitter:

    def __init__(self, model, layer_name):
        self.layers = get_model_layers(model, layer_name=layer_name)
        self.num_layers = len(self.layers)
        self.num_layer_components = len(tuple(self.layers[0].parameters()))


def get_train_params(
    model: nn.Module,
    layer_low_id: int = None,
    layer_mid_id: int = None,
    head_param_start: int = None,
    embed_lr: float = None,
    base_lr: float = 2e-5,
    head_lr: float = None,
    weight_decay: float = 1e-2,
    low_ratio: float = 0.4,
    high_ratio: float = 2.,
    gaussian_lr: bool = False,
    gaussian_sigs: float = 0.5,
    layer_name=None,
    min_lr_scale: float = 0.1,
    group_lr_dict: dict = None,
    filter_freeze=False
):
    num_layer_components = 1 if layer_name is None else ModelLayerFitter(model, layer_name).num_layer_components

    named_parameters = list(model.named_parameters())

    if head_lr is None:
        head_lr = base_lr
    if embed_lr is None:
        embed_lr = base_lr

    if head_param_start is None:
        head_param_start = len(named_parameters) - 2

    embed_parameters = named_parameters[0:1]
    embed_group = [params for (_, params) in embed_parameters]
    backbone_parameters = named_parameters[1:head_param_start]
    head_parameters = named_parameters[head_param_start:]
    head_group = [params for (_, params) in head_parameters]

    parameters = []
    if embed_lr > 0.:
        parameters.append({'params': embed_group, 'lr': embed_lr})
    else:
        for params in embed_group:
            params.requires_grad = False
        if not filter_freeze:
            parameters.append({'params': embed_group, 'lr': embed_lr})
    if head_lr > 0.:
        parameters.append({'params': head_group, 'lr': head_lr})
    else:
        for params in head_group:
            params.requires_grad = False
        if not filter_freeze:
            parameters.append({'params': embed_group, 'lr': embed_lr})

    if layer_low_id is None:
        layer_low_id = 0
    if layer_mid_id is None:
        layer_mid_id = head_param_start

    n = len(backbone_parameters) // num_layer_components
    mu = n / 2
    sig = mu / gaussian_sigs
    peak = 1 - min_lr_scale
    g = Gaussian(mu=mu, sig=sig, peak=peak)
    g0 = g(0)
    s = peak / (peak - g0)

    for i, (name, params) in enumerate(backbone_parameters):
        if not params.requires_grad and filter_freeze:
            continue
        weight_decay: float = 0.0 if ('bias' in name) or ('LayerNorm' in name) or ('RMSNorm' in name) else weight_decay
        layer_num = i // num_layer_components

        if gaussian_lr:
            lr = (1. - s*(g(layer_num+1)-g0)) * base_lr
        else:
            lr = base_lr
        if group_lr_dict:
            for k, l in group_lr_dict.items():
                if k in name:
                    lr *= l
                    break
        if layer_low_id is not None and layer_mid_id is not None:
            if layer_num >= layer_mid_id:
                lr *= high_ratio
            elif layer_num < layer_low_id:
                lr *= low_ratio

        if lr > 0.:
            parameters.append({'params': params, 'weight_decay': weight_decay, 'lr': lr})
        else:
            params.requires_grad = False
            if not filter_freeze:
                parameters.append({'params': params, 'weight_decay': weight_decay, 'lr': lr})

    return parameters
