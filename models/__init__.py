from __future__ import absolute_import

from models.net import Net

__factory = {
    'Net': Net,
}

def get_names():
    return __factory.keys()

def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown models: {}".format(name))
    return __factory[name](*args, **kwargs)
