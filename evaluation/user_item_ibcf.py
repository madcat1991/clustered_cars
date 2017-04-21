"""
The experiment with user-item (only information about users and items) IBCF
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
    with open(args.top_k_per_uid) as f:
        top_k_per_uid = pickle.load(f)

    logging.info("Building recommendations using user-specific top-ks")
    ui_iid_recs = {}
    col_to_iid = {col_id: iid for iid, col_id in iid_to_col.items()}
    for t in testing_df.itertuples():
        key = (t.code, t.propcode)

        row_id = uid_to_row[t.code]
        top_k = top_k_per_uid[key]

        if row_id is not None:
            rec_row = recs_m[row_id]

            iid_recs = []
            for arg_id in np.argsort(rec_row.data)[-top_k:][::-1]:
                iid = col_to_iid[rec_row.indices[arg_id]]
                iid_recs.append(iid)
                if t.propcode == iid:
                    break

            ui_iid_recs[key] = iid_recs

    logging.info("Storing users' recommendations to: %s", args.ui_recs_path)
    with open(args.ui_recs_path, "w") as f:
        pickle.dump(ui_iid_recs, f)


def main():
    logging.info("Reading training data")
    training_df = pd.read_csv(args.training_csv)
    tr_m, uid_to_row, iid_to_col = get_training_matrix_and_indices(training_df)
    logging.info("Training matrix: %s", get_sparse_matrix_info(tr_m))

    logging.info("Reading testing data")
    testing_df = pd.read_csv(args.testing_csv)[["code", "propcode"]].drop_duplicates()

    logging.info("Preparing similarity matrix")
    sim_m = get_similarity_matrix(tr_m)

    logging.info("Testing hit ratio at top-%s", args.top_k)
    recs_m = get_topk_recs(
        normalize(tr_m),
        sim_m,
        binarize(tr_m),
        args.top_k,
    )
    logging.info("Hit ratio: %.3f", hit_ratio(recs_m, testing_df, uid_to_row, iid_to_col))

    if args.top_k_per_uid:
        recs_m = get_topk_recs(
            tr_m,
            sim_m,
            binarize(tr_m)
        )
        store_data_for_eval(recs_m, testing_df, uid_to_row, iid_to_col)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-k", default=20, type=int, dest="top_k",
                        help="Number of recommended items per a user. Default: 20")
    parser.add_argument("--trf", default='training.csv', dest="training_csv",
                        help="Training data file name. Default: training.csv")
    parser.add_argument("--tsf", default='testing.csv', dest="testing_csv",
                        help="Testing data file name. Default: testing.csv")

    parser.add_argument("--ek", dest="top_k_per_uid",
                        help="Path to the *.pkl containing the value of top-k per each user (from ug_bg_ibcf.py). "
                             "If specified, then the resulting recommendation per each user are stored to --er")
    parser.add_argument("--er", default="ui_iid_recs.pkl", dest="ui_recs_path",
                        help="Path to the file to store users recommendations for evaluation. Check --ek. "
                             "Default: ui_iid_recs.pkl")

    parser.add_argument("--log-level", default='INFO', dest="log_level",
                        choices=['DEBUG', 'INFO', 'WARNINGS', 'ERROR'], help="Logging level")

    args = parser.parse_args()
    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s', stream=sys.stdout, level=getattr(logging, args.log_level)
    )

    main()
