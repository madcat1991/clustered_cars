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

from ibcf.matrix_functions import get_sparse_matrix_info
from ibcf.recs import get_topk_recs
from ibcf.similarity import get_similarity_matrix
from misc.common import get_ug_data, get_bg_data


def get_matrix(df, uid_to_ug, bid_to_bg):
    uids_per_ug = pd.Series(list(uid_to_ug.values())).value_counts()
    u_mult = 1.0 / uids_per_ug.sort_index().values

    bids_per_ug = pd.Series(list(bid_to_bg.values())).value_counts()
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


def print_recs(recs_m, user_list, uid_to_ug):
    for uid in user_list:
        ug = uid_to_ug.get(uid)
        if ug is not None:
            rec_row = recs_m[ug]
            rec_bgs = []
            for arg_id in np.argsort(rec_row.data)[::-1]:
                bg = rec_row.indices[arg_id]
                rec_bgs.append(bg)
            print("User %s, cluster #%s: %s" % (uid, ug, ", ".join(map(str, rec_bgs))))
        else:
            print("User %s: does not exist in the dataset" % uid)


def main():
    logging.info(u"Getting clusters data")
    uid_to_ug = get_ug_data(args.user_cluster)
    bid_to_bg, bg_iids = get_bg_data(args.booking_cluster)

    logging.info(u"Creating user-item matrix")
    df = pd.read_csv(args.data_csv)
    ui_m = get_matrix(df, uid_to_ug, bid_to_bg)
    logging.info(u"Training matrix: %s", get_sparse_matrix_info(ui_m))

    sim_m = get_similarity_matrix(ui_m)
    recs_m = get_topk_recs(
        ui_m,
        sim_m,
        binarize(ui_m),
        3
    )
    logging.info(u"Recs are ready")

    random_users = [
        'NMS730912', 'NMS741634', 'NMS549598', 'HH1995', 'NMS982618', 'NMS1013340', 'ZY25701',
        'NMS771535', 'NMS395390', 'ZZ35721', 'NMS61584', 'HM689', 'NMS585593', 'NMS981927',
        'NMS1066637', 'NMS1029908', 'NMS631625', 'NMS276860', 'NMS679126', 'NMS268544'
    ]
    print("\nRecs for random users:")
    print_recs(recs_m, random_users, uid_to_ug)

    selected_users = [
        'NMS449617', 'ZY82553', 'NMS230443', 'NMS105985', 'HS14266', 'NMS178887',
        'HF9739', 'NMS316975', 'NMS590280'
    ]
    print("\nRecs for selected users:")
    print_recs(recs_m, selected_users, uid_to_ug)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-d", default='booking.csv', dest="data_csv",
                        help=u"Path to the file with bookings. Default: booking.csv")
    parser.add_argument("-u", default='users.txt', dest="user_cluster",
                        help=u"Path to the file with user clusters. Default: users.txt")
    parser.add_argument("-b", default='bookings.txt', dest="booking_cluster",
                        help=u"Path to the file with booking clusters. Default: bookings.txt")
    parser.add_argument("--log-level", default='INFO', dest="log_level",
                        choices=['DEBUG', 'INFO', 'WARNINGS', 'ERROR'], help=u"Logging level")

    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s', stream=sys.stdout, level=getattr(logging, args.log_level)
    )

    main()
