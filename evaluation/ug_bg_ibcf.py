"""
The experiment with user cluster - booking cluster IBCF.
"""

import argparse
import logging
import pickle
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import binarize

from ibcf.matrix_functions import get_sparse_matrix_info
from ibcf.recs import get_topk_recs
from ibcf.similarity import get_similarity_matrix
from misc.common import get_ug_data, get_bg_data
from model.build_recs_matrix import get_matrix


def hit_ratio(recs_m, testing_df, uid_to_ug, bg_iids):
    hit = 0

    logging.info("# of testing instances: %s", testing_df.shape[0])
    for t in testing_df.itertuples():
        row_id = uid_to_ug.get(t.code)

        if row_id is not None:
            rec_row = recs_m[row_id]

            for arg_id in np.argsort(rec_row.data)[::-1]:
                bg_id = rec_row.indices[arg_id]
                if t.propcode in bg_iids[bg_id]:
                    hit += 1
                    break

    return hit / testing_df.shape[0]


def store_data_for_eval(recs_m, testing_df, uid_to_ug, bg_iids):
    ui_bg_recs = {}
    ui_iids_cnt = Counter()

    logging.info("Finding the number of items that should be processed before the test item will be find")
    for t in testing_df.itertuples():
        key = (t.code, t.propcode)
        row_id = uid_to_ug.get(t.code)

        if row_id is not None:
            rec_row = recs_m[row_id]

            bg_recs = []
            processed_items = set()

            for arg_id in np.argsort(rec_row.data)[::-1]:
                bg_id = rec_row.indices[arg_id]
                bg_recs.append(bg_id)

                if t.propcode in bg_iids[bg_id]:
                    break

                processed_items.update(bg_iids[bg_id])

            ui_iids_cnt[key] = len(processed_items) + 1  # +1 is the break position
            ui_bg_recs[key] = bg_recs

    logging.info("Storing the found top-k numbers to: %s", args.top_k_iid_per_uid)
    with open(args.top_k_iid_per_uid, "wb") as f:
        pickle.dump(ui_iids_cnt, f)

    logging.info("Storing users' bg recommendations without top-k to: %s", args.ui_bg_recs_path)
    with open(args.ui_bg_recs_path, "wb") as f:
        pickle.dump(ui_bg_recs, f)


def main():
    logging.info(u"Getting clusters data")
    uid_to_ug = get_ug_data(args.user_cluster)
    bid_to_bgs, bg_iids = get_bg_data(args.booking_cluster)

    logging.info("Reading training data")
    training_df = pd.read_csv(args.training_csv)
    tr_m = get_matrix(training_df, uid_to_ug, bid_to_bg)
    logging.info(u"Training matrix: %s", get_sparse_matrix_info(tr_m))

    logging.info("Reading testing data")
    # we don't care about repetitive actions in the testing
    testing_df = pd.read_csv(args.testing_csv)[["code", "propcode"]].drop_duplicates()

    logging.info("Preparing similarity matrix")
    sim_m = get_similarity_matrix(tr_m)

    logging.info("Testing hit ratio at top-%s", args.top_k)
    recs_m = get_topk_recs(
        tr_m,
        sim_m,
        binarize(tr_m),
        args.top_k
    )
    logging.info(u"Hit ratio: %.3f", hit_ratio(recs_m, testing_df, uid_to_ug, bg_iids))

    if args.top_k_iid_per_uid:
        recs_m = get_topk_recs(
            tr_m,
            sim_m,
            binarize(tr_m)
        )
        store_data_for_eval(recs_m, testing_df, uid_to_ug, bg_iids)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-u", required=True, dest="user_cluster",
                        help=u"Path to the file containing information about user clusters")
    parser.add_argument("-b", required=True, dest="booking_cluster",
                        help=u"Path to the file containing information about booking clusters")
    parser.add_argument("-k", default=3, type=int, dest="top_k",
                        help="Number of recommended booking clusters per a user. Default: 3")

    parser.add_argument("--trf", default='training.csv', dest="training_csv",
                        help=u"Training data file name. Default: training.csv")
    parser.add_argument("--tsf", default='testing.csv', dest="testing_csv",
                        help=u"Testing data file name. Default: testing.csv")

    parser.add_argument("--ek", dest="top_k_iid_per_uid",
                        help="Path to the *.pkl where to save the value of top-k items per each user. "
                             "If specified, then the resulting recommendation per each user are stored to --er")
    parser.add_argument("--er", default="ui_bg_recs.pkl", dest="ui_bg_recs_path",
                        help="Path to the file to store users recommendations for evaluation. Check --ek. "
                             "Default: ui_bg_recs.pkl")

    parser.add_argument("--log-level", default='INFO', dest="log_level",
                        choices=['DEBUG', 'INFO', 'WARNINGS', 'ERROR'], help=u"Logging level")

    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s', stream=sys.stdout, level=getattr(logging, args.log_level)
    )

    main()
