import os

import numpy as np


def get_abs_path(*args):
    """ Concatenates path's parts sent through args and returns the
        absolute path to the file
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), *args))


def clean_json_dict_keys(d):
    new_d = {}
    for k, v in d.items():
        _k = int(k) if isinstance(k, np.int32) else k
        new_d[_k] = clean_json_dict_keys(v) if isinstance(v, dict) else v
    return new_d
