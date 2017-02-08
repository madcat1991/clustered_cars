# coding: utf-8

import numpy as np
import pandas as pd
from pandas.tools.tile import _bins_to_cuts


def _fix_outliers(s, min_v, max_v):
    s[pd.notnull(s) & (s < min_v)] = min_v
    s[pd.notnull(s) & (s > max_v)] = max_v
    return s


def _density_based_cutter(s, min_bin_size, min_value, max_value):
    hist, bin_edges = np.histogram(s, bins='doane', range=(min_value, max_value))
    hist[hist < min_bin_size] = 0
    pos_ids = np.where(hist > 0)[0]

    bin_edges = bin_edges[pos_ids]
    bin_edges[0] = min(bin_edges[0], min_value)
    if bin_edges[-1] < max_value:
        bin_edges = np.append(bin_edges, max_value)

    return _bins_to_cuts(s, bin_edges, include_lowest=True, retbins=True)


def prepare_num_column(s, max_value_p=99.95, min_bin_p=0.05, min_value=None):
    s = _fix_outliers(s, 0, np.percentile(s, max_value_p))

    min_bin_size = s.size * min_bin_p
    if min_value is None:
        min_value = s.min()

    return _density_based_cutter(s, min_bin_size, min_value, s.max())[0]
