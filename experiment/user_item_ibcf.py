# coding: utf-8

u"""
The experiment with user-item IBCF. Only information about users and items
"""

import argparse
import logging
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import binarize
from sklearn.preprocessing import normalize

from ibcf.matrix_functions import get_sparse_matrix_info
from ibcf.recs import get_topk_recs
from ibcf.similarity import get_similarity_matrix


def get_training_matrix_and_indices(df):
    cols = []
    rows = []

    uid_to_row = {}
    iid_to_col = {}

    for t in df.itertuples():
        row_id = uid_to_row.setdefault(t.code, len(uid_to_row))
        col_id = iid_to_col.setdefault(t.propcode, len(iid_to_col))

        rows.append(row_id)
        cols.append(col_id)

    m = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(uid_to_row), len(iid_to_col)))
    return m, uid_to_row, iid_to_col


def get_testing_matrix(df, uid_to_row, iid_to_col):
    cols = []
    rows = []

    for t in df.itertuples():
        row_id = uid_to_row.get(t.code)
        col_id = iid_to_col.get(t.propcode)

        if row_id is not None and col_id is not None:
            rows.append(row_id)
            cols.append(col_id)

    m = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(uid_to_row), len(iid_to_col)))
    return m


def hit_ratio(recs_m, ts_m):
    res = recs_m.multiply(ts_m)
    return res.nnz / float(ts_m.nnz)


def main():
    training_df = pd.read_csv(args.training_csv)
    tr_m, uid_to_row, iid_to_col = get_training_matrix_and_indices(training_df)
    logging.info(u"Training matrix: %s", get_sparse_matrix_info(tr_m))

    testing_df = pd.read_csv(args.testing_csv)
    ts_m = get_testing_matrix(testing_df, uid_to_row, iid_to_col)
    logging.info(u"Testing matrix: %s", get_sparse_matrix_info(ts_m))

    sim_m = get_similarity_matrix(tr_m)
    recs_m = get_topk_recs(
        normalize(tr_m),
        sim_m,
        binarize(tr_m),
        10
    )
    logging.info(u"Hit ratio: %.3f", hit_ratio(recs_m, ts_m))


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    # parser.add_argument("--trf", default='training.csv', dest="training_csv",
    #                     help=u"Training data file name. Default: training.csv")
    # parser.add_argument("--tsf", default='testing.csv', dest="testing_csv",
    #                     help=u"Testing data file name. Default: testing.csv")
    # parser.add_argument("--log-level", default='INFO', dest="log_level",
    #                     choices=['DEBUG', 'INFO', 'WARNINGS', 'ERROR'], help=u"Logging level")
    #
    # args = parser.parse_args()

    from collections import namedtuple
    args = namedtuple("args", ["training_csv", "testing_csv", "log_level"])

    args.training_csv = '/Users/user/PyProjects/clustered_cars/data/training.csv'
    args.testing_csv = '/Users/user/PyProjects/clustered_cars/data/testing.csv'
    args.log_level = 'INFO'

    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s', stream=sys.stdout, level=getattr(logging, args.log_level)
    )

    main()
