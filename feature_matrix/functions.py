import logging
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def fix_outliers(col_data, min_val=None, max_val=None, return_min_max=False):
    non_na_data = col_data.dropna()

    # removing outliers
    if min_val is None:
        min_val = np.percentile(non_na_data, 1)
    if max_val is None:
        max_val = np.percentile(non_na_data, 99)

    col_data[pd.notnull(col_data) & (col_data < min_val)] = min_val
    col_data[pd.notnull(col_data) & (col_data > max_val)] = max_val

    if return_min_max:
        return col_data, min_val, max_val
    return col_data


def get_bins_for_num_column(col_data, n_bins):
    x = col_data.dropna().values
    km = KMeans(n_clusters=n_bins).fit(x.reshape(-1, 1))

    bins = [min(x)]
    is_bad_edge = False
    for cl_id in np.argsort(km.cluster_centers_.reshape(-1)):
        if is_bad_edge:
            prev_edge = (bins[-1] + min(x[km.labels_ == cl_id])) / 2
            bins.append(prev_edge)

        edge = max(x[km.labels_ == cl_id])
        is_bad_edge = bins[-1] == edge

        if not is_bad_edge:
            bins.append(edge)
    return bins


def replace_numerical_to_categorical(df, binning_cols, min_p=0.005, max_p=0.995):
    logging.info(u"Binning DF numerical columns to categorical: %s", binning_cols)
    for col, n_bins in binning_cols.items():
        logging.info("Binning columns '%s'", col)
        col_data = fix_outliers(df[col].copy())
        bins = get_bins_for_num_column(col_data, n_bins)
        col_data = pd.cut(col_data, bins, include_lowest=True).astype('object')

        val_count = col_data.dropna().value_counts() / float(col_data.shape[0])
        values_to_remove = val_count[(val_count < min_p) | (val_count > max_p)].index
        df[col] = col_data.replace(values_to_remove, None)
    return df
