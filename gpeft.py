import random

from .gfft import get_model_layers
from .utils import Gaussian


def gpeft(model, max_p=0.8, layer_name='model.layers', sigs=3):
    """
    After model loaded:
        model = AutoModelForCausalLM.from_pretrained('...')
        gpeft(model)
        optimizer = Optimizer(model.parameters(), lr=lr, ...)
    """
    layers = get_model_layers(model, layer_name)
    n = len(layers)
    mu = n / 2
    sig = mu / sigs
    g = Gaussian(mu=mu, sig=sig, peak=max_p)
    for i, layer in enumerate(layers, 1):
        p = g(i)
        if random.random() < p:
            for name, params in layer.named_parameters():
                params.requires_grad = False
