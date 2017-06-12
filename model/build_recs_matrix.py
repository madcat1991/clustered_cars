"""
This script builds the matrix of recommendations of booking clusters to
user clusters
"""

import argparse
import logging
import sys

import numpy as np
import pandas as pd
from scipy.io import mmwrite
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from ibcf.matrix_functions import get_sparse_matrix_info
from ibcf.recs import get_topk_recs
from ibcf.similarity import get_similarity_matrix
from misc.common import get_ug_data, get_bg_data


def get_matrix(df, uid_to_ug, bid_to_bg):
    """Creates a matrix of the probabilities that a user uid from uid_to_ug
    will book a booking bid from bid_to_bg given the number of times
    ug has been observed with bg.

    :param df: data frame containing information about bookings
    :param uid_to_ug: uid -> ug index
    :param bid_to_bg: bid -> bg index
    :return: a matrix of probabilities
    """
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

    # probability to observe bg by ug
    m = normalize(m, norm='l1')
    # + probability to observe uid in ug
    m = csr_matrix(m.multiply(u_mult.reshape(-1, 1)))
    # + probability to observe bid in bg
    m = csr_matrix(m.multiply(b_mult))
    return m


def main():
    logging.info(u"Loading clustered data")
    uid_to_ug = get_ug_data(args.user_cluster)
    bid_to_bg, _ = get_bg_data(args.booking_cluster)

    logging.info(u"Building ug-bg matrix")
    df = pd.read_csv(args.data_csv)
    ui_m = get_matrix(df, uid_to_ug, bid_to_bg)
    logging.info(u"Training matrix: %s", get_sparse_matrix_info(ui_m))

    logging.info(u"Building similarity matrix")
    sim_m = get_similarity_matrix(ui_m)

    logging.info(u"Building ug-bg recs matrix")
    recs_m = get_topk_recs(ui_m, sim_m)

    logging.info(u"Dumping recs matrix")
    mmwrite(args.recs_path, recs_m)
    logging.info("Finish")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-d", default='booking.csv', dest="data_csv",
                        help=u"Path to the file with bookings. Default: booking.csv")
    parser.add_argument("-u", default='users.txt', dest="user_cluster",
                        help=u"Path to the file with user clusters. Default: users.txt")
    parser.add_argument("-b", default='bookings.txt', dest="booking_cluster",
                        help=u"Path to the file with booking clusters. Default: bookings.txt")
    parser.add_argument("-o", default='ug_bg_recs.mtx', dest="recs_path",
                        help=u"Path to the output file for the recommendation matrix. "
                             u"Default: ug_bg_recs.mtx")
    parser.add_argument("--log-level", default='INFO', dest="log_level",
                        choices=['DEBUG', 'INFO', 'WARNINGS', 'ERROR'], help=u"Logging level")

    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s', stream=sys.stdout, level=getattr(logging, args.log_level)
    )

    main()
