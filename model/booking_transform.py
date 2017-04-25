"""
The script transforms bookings data into binary mode.
"""

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from preprocessing.common import canonize_datetime, check_processed_columns
from feature_matrix.functions import density_based_cutter, fix_outliers

COLS_TO_BIN = ["adults", "children", "babies", "avg_spend_per_head", "n_booked_days", "drivetime"]
RESERVED_COLS = ["code", "year", "propcode", "bookcode"]
DATE_COLS = [u'bookdate', u'sdate', u"fdate"]
COLS_TO_DROP = [
    u'bookdate',  # no need
    u'sourcedesc'  # too detailed
]


def actualize_testing_data(training_df, testing_df):
    logging.info(u"Testing data, before cleaning: %s", testing_df.shape)
    # removing from testing unknown users and items
    known_user_ids = training_df.code.unique()
    known_item_ids = training_df.propcode.unique()
    testing_df = testing_df[
        testing_df.code.isin(known_user_ids) & testing_df.propcode.isin(known_item_ids)
    ]

    # removing from testing user/item pairs that have been seen in training
    known_pairs = {(t.code, t.propcode) for t in training_df.itertuples()}
    mask = [(t.code, t.propcode) not in known_pairs for t in testing_df.itertuples()]
    testing_df = testing_df[mask]
    logging.info(u"Testing data, after cleaning: %s", testing_df.shape)
    return testing_df


def _clean_numeric_outliers(bdf, max_p=99.9):
    for col in COLS_TO_BIN:
        max_value = np.percentile(bdf[col], max_p)
        bdf[col] = fix_outliers(bdf[col], 0, max_value)
    return bdf


def _convert_to_cat(bdf, col, bins):
    s, bin_edges = density_based_cutter(bdf[col], bins)
    bdf[col] = s.astype('object')
    return bdf


def _numeric_to_categorical(bdf):
    n_bins_per_col = {
        "adults": 4,
        "children": 3,
        "babies": 3,
        "avg_spend_per_head": 5,
        "n_booked_days": 5,
        "drivetime": 4,
    }

    assert not set(n_bins_per_col).difference(COLS_TO_BIN)

    for col, n_bin in n_bins_per_col.items():
        bdf = _convert_to_cat(bdf, col, n_bin)

    return bdf


def _clean_categorical_outliers(bdf, min_p=0.005, max_p=0.995):
    for col in bdf.columns[bdf.dtypes == 'object'].drop(RESERVED_COLS, 'ignore'):
        val_count = bdf[col].dropna().value_counts() / float(bdf.shape[0])
        values_to_remove = val_count[(val_count < min_p) | (val_count > max_p)].index
        bdf[col] = bdf[col].replace(values_to_remove, None)
    return bdf


def remove_unrepresentative_users(bdf, min_bookings_per_user):
    if min_bookings_per_user > 1:
        logging.info("DF, before cleaning: %s", bdf.shape)
        bookings_per_user = bdf.code.value_counts()
        logging.info("Removing users having less than %s bookings", min_bookings_per_user)
        good_user_ids = bookings_per_user[bookings_per_user >= min_bookings_per_user].index
        bdf = bdf[bdf.code.isin(good_user_ids)]
        logging.info("DF data, after cleaning: %s", bdf.shape)
    return bdf


def prepare_for_categorization(bdf):
    bdf = bdf.drop(COLS_TO_DROP, axis=1)
    bdf[u'n_booked_days'] = (bdf.fdate - bdf.sdate).apply(lambda x: x.days)
    bdf = bdf.drop([u'sdate', u'fdate'], axis=1)
    bdf.drivetime = np.round(bdf.drivetime / 3600)  # to hours
    bdf.n_booked_days = bdf.n_booked_days.apply(lambda x: 1 if pd.isnull(x) or x < 1 else x)
    bdf.avg_spend_per_head /= bdf.n_booked_days.astype(float)
    return bdf


def replace_numerical_to_categorical(bdf):
    logging.info(u"Converting DF to categorical columns, shape: %s", bdf.shape)
    bdf = _clean_numeric_outliers(bdf)
    bdf = _numeric_to_categorical(bdf)
    bdf = _clean_categorical_outliers(bdf)
    logging.info(u"Converted DF shape: %s", bdf.shape)
    return bdf


def main():
    logging.info(u"Start")
    bdf = pd.read_csv(args.data_csv_path)
    bdf = canonize_datetime(bdf, DATE_COLS)
    original_columns = bdf.columns

    # categorizing
    bdf = remove_unrepresentative_users(bdf, args.min_bookings_per_user)
    bdf = prepare_for_categorization(bdf)
    bdf = replace_numerical_to_categorical(bdf)

    # quality check
    columns = set(bdf.columns).union(COLS_TO_DROP + DATE_COLS).difference(['n_booked_days'])
    check_processed_columns(columns, original_columns)

    bdf.to_csv(args.output_path, index=False)
    logging.info(u"Finish")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-d", required=True, dest="data_csv_path",
                        help=u"Path to a csv file with the cleaned bookings")
    parser.add_argument("-m", dest="min_bookings_per_user", type=int, default=1,
                        help=u"Min bookings per user. Default: 1")
    parser.add_argument("-o", default='t_bookings.csv', dest="output_path",
                        help=u"Path to the output CSV. Default: t_bookings.csv")
    parser.add_argument("--log-level", default='INFO', dest="log_level",
                        choices=['DEBUG', 'INFO', 'WARNINGS', 'ERROR'], help=u"Logging level")

    args = parser.parse_args()
    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s', stream=sys.stdout, level=getattr(logging, args.log_level)
    )

    main()
