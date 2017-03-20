# coding: utf-8

import numpy as np
import pandas as pd
from pandas.tools.tile import _bins_to_cuts


def fix_outliers(s, min_v, max_v):
    s[pd.notnull(s) & (s < min_v)] = min_v
    s[pd.notnull(s) & (s > max_v)] = max_v
    return s


def density_based_cutter(s, bins):
    min_value = max(1, int(s.min()))
    max_value = int(s.max())

    bin_edges = list(range(min_value, max_value + 1))
    hist = [0] * (len(bin_edges))

    for value in s:
        hist[int(value) - min_value] += 1

    while len(bin_edges) - 1 > bins:
        target_id = np.argmin(hist)

        if target_id != 0 and target_id != len(hist) - 1:
            if hist[target_id - 1] < hist[target_id + 1]:
                hist[target_id - 1] += hist[target_id]
                bin_edges.pop(target_id)
                hist.pop(target_id)
            else:
                if target_id + 1 != len(hist) - 1:
                    hist[target_id] += hist[target_id + 1]
                    bin_edges.pop(target_id + 1)
                    hist.pop(target_id + 1)
                else:
                    hist[target_id] += hist[target_id - 1]
                    bin_edges.pop(target_id - 1)
                    hist.pop(target_id - 1)
        elif target_id == 0:
            hist[target_id] += hist[target_id + 1]
            bin_edges.pop(target_id + 1)
            hist.pop(target_id + 1)
        else:
            hist[target_id] += hist[target_id - 1]
            bin_edges.pop(target_id - 1)
            hist.pop(target_id - 1)
    return _bins_to_cuts(s, np.array(bin_edges), include_lowest=True, retbins=True)
