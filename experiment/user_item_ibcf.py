"""
The experiment with user-item IBCF. Only information about users and items
"""

import argparse
import logging
import pickle
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
    m = binarize(m)  # we don't care about repetitive actions in the testing
    return m


def hit_ratio(recs_m, testing_df, uid_to_row, iid_to_col):
    hit = 0
    for t in testing_df.itertuples():
        row_id = uid_to_row[t.code]
        col_id = iid_to_col[t.propcode]

        if row_id is not None:
            rec_row = recs_m[row_id]
            rec_cols = {rec_row.indices[arg_id] for arg_id in np.argsort(rec_row.data)[::-1]}

            if col_id in rec_cols:
                hit += 1
    return float(hit) / testing_df.shape[0]


def store_data_for_eval(recs_m, testing_df, uid_to_row, iid_to_col):
    col_to_iid = {v: k for k, v in iid_to_col.items()}
    with open("ui_iids_cnt.pkl") as f:
        ui_iids_cnt = pickle.load(f)

    ui_iid_recs = {}
    for t in testing_df.itertuples():
        key = (t.code, t.propcode)

        row_id = uid_to_row[t.code]
        top = ui_iids_cnt[key]

        if row_id is not None:
            rec_row = recs_m[row_id]

            iid_recs = []
            for arg_id in np.argsort(rec_row.data)[-top:][::-1]:
                iid = col_to_iid[rec_row.indices[arg_id]]
                iid_recs.append(iid)
                if t.propcode == iid:
                    break

            ui_iid_recs[key] = iid_recs

    with open("ui_iid_recs.pkl", "w") as f:
        pickle.dump(ui_iid_recs, f)


def main():
    training_df = pd.read_csv(args.training_csv)
    tr_m, uid_to_row, iid_to_col = get_training_matrix_and_indices(training_df)
    logging.info(u"Training matrix: %s", get_sparse_matrix_info(tr_m))

    testing_df = pd.read_csv(args.testing_csv)[["code", "propcode"]].drop_duplicates()

    sim_m = get_similarity_matrix(tr_m)
    recs_m = get_topk_recs(
        normalize(tr_m),
        sim_m,
        binarize(tr_m),
        20,
    )
    logging.info(u"Hit ratio: %.3f", hit_ratio(recs_m, testing_df, uid_to_row, iid_to_col))

    recs_m = get_topk_recs(
        tr_m,
        sim_m,
        binarize(tr_m)
    )
    store_data_for_eval(recs_m, testing_df, uid_to_row, iid_to_col)


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
