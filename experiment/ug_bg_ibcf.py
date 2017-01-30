# coding: utf-8

u"""
The experiment with user cluster - booking cluster IBCF.
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


def get_training_matrix(df, uid_to_ug, bid_to_bg):
    uids_per_ug = pd.Series(uid_to_ug.values()).value_counts()
    u_mult = 1.0 / uids_per_ug.sort_index().values

    bids_per_ug = pd.Series(bid_to_bg.values()).value_counts()
    b_mult = 1.0 / bids_per_ug.sort_index().values

    cols = []
    rows = []
    for t in df.itertuples():
        rows.append(uid_to_ug[t.code])
        cols.append(bid_to_bg[t.bookcode])

    m = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(u_mult), len(b_mult)))
    m = csr_matrix(m.multiply(u_mult.reshape(u_mult.shape[0], 1)))
    m = csr_matrix(m.multiply(b_mult))
    return m


def hit_ratio(recs_m, testing_df, uid_to_ug, bg_iids):
    hit = 0

    from collections import Counter
    counter = Counter()

    for t in testing_df.itertuples():
        row_id = uid_to_ug.get(t.code)

        if row_id is not None:
            rec_row = recs_m[row_id]

            for arg_id in np.argsort(rec_row.data)[::-1]:
                bg_id = rec_row.indices[arg_id]
                if t.propcode in bg_iids[bg_id]:
                    hit += 1
                    break

                counter[(t.code, t.propcode)] += len(bg_iids[bg_id])
            else:
                del counter[(t.code, t.propcode)]

    import pickle
    with open("hits.pkl", "w") as f:
        pickle.dump(counter, f)

    return float(hit) / testing_df.shape[0]


def get_ug_data():
    uid_to_ug = {}
    with open(args.user_cluster) as f:
        # skipping
        while not f.next().startswith("Cluster"):
            pass

        cl_id = 0
        for line in f:
            if line.startswith("Users:"):
                for uid in line.lstrip("Users:").split(","):
                    uid_to_ug[uid.strip()] = cl_id
            elif line.startswith("Cluster"):
                cl_id += 1
    return uid_to_ug


def get_bg_data():
    bid_to_bg = {}
    bg_iids = {}
    with open(args.booking_cluster) as f:
        # skipping
        while not f.next().startswith("Cluster"):
            pass

        cl_id = 0
        for line in f:
            if line.startswith("Bookings:"):
                for bid in line.lstrip("Bookings:").split(","):
                    bid_to_bg[bid.strip()] = cl_id
            elif line.startswith("Items:"):
                bg_iids[cl_id] = {iid.strip() for iid in line.lstrip("Items:").split(",")}
            elif line.startswith("Cluster"):
                cl_id += 1
    return bid_to_bg, bg_iids


def main():
    logging.info(u"Getting clusters data")
    uid_to_ug = get_ug_data()
    bid_to_bg, bg_iids = get_bg_data()

    logging.info(u"Creating matrices")
    training_df = pd.read_csv(args.training_csv)
    tr_m = get_training_matrix(training_df, uid_to_ug, bid_to_bg)
    logging.info(u"Training matrix: %s", get_sparse_matrix_info(tr_m))

    # we don't care about repetitive actions in the testing
    testing_df = pd.read_csv(args.testing_csv)[["code", "propcode"]].drop_duplicates()

    sim_m = get_similarity_matrix(tr_m)
    recs_m = get_topk_recs(
        tr_m,
        sim_m,
        binarize(tr_m),
        3
    )
    logging.info(u"Hit ratio: %.3f", hit_ratio(recs_m, testing_df, uid_to_ug, bg_iids))


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
    args = namedtuple(
        "args",
        ["training_csv", "testing_csv", "booking_cluster", "user_cluster", "log_level"]
    )

    args.training_csv = '/Users/user/PyProjects/clustered_cars/data/training.csv'
    args.testing_csv = '/Users/user/PyProjects/clustered_cars/data/testing.csv'
    args.user_cluster = '/Users/user/PyProjects/clustered_cars/data/clustered/user.txt'
    args.booking_cluster = '/Users/user/PyProjects/clustered_cars/data/clustered/booking.txt'
    args.log_level = 'INFO'

    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s', stream=sys.stdout, level=getattr(logging, args.log_level)
    )

    main()