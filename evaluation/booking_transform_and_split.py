"""
The script splits a booking csv dataset into testing and training
parts based on timestamp column. Then it transforms the parts
into binary mode.
"""

import argparse
import logging
import sys

import numpy as np
import pandas as pd
from pandas.tools.tile import _bins_to_cuts

from model.booking_transform import remove_unrepresentative_users, prepare_for_categorization, \
    RESERVED_COLS, DATE_COLS, COLS_TO_DROP, COLS_TO_BIN
from preprocessing.common import canonize_datetime, check_processed_columns
from feature_matrix.functions import density_based_cutter, fix_outliers
from misc.splitter import TimeWindowSplitter


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


def _clean_numeric_outliers(training_df, testing_df, max_p=99.9):
    for col in COLS_TO_BIN:
        max_value = np.percentile(training_df[col], max_p)
        training_df[col] = fix_outliers(training_df[col], 0, max_value)
        testing_df[col] = fix_outliers(testing_df[col], 0, max_value)
    return training_df, testing_df


def _convert_to_cat(training_df, testing_df, col, bins):
    s, bin_edges = density_based_cutter(training_df[col], bins)
    training_df[col] = s
    testing_df[col] = _bins_to_cuts(testing_df[col], bin_edges, include_lowest=True)
    training_df[col] = training_df[col].astype('object')
    testing_df[col] = testing_df[col].astype('object')
    return training_df, testing_df


def _numeric_to_categorical(training_df, testing_df):
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
        training_df, testing_df = _convert_to_cat(training_df, testing_df, col, n_bin)

    return training_df, testing_df


def _clean_categorical_outliers(training_df, testing_df, min_p=0.005, max_p=0.995):
    for col in training_df.columns[training_df.dtypes == 'object'].drop(RESERVED_COLS, 'ignore'):
        val_count = training_df[col].dropna().value_counts() / float(training_df.shape[0])
        values_to_remove = val_count[(val_count < min_p) | (val_count > max_p)].index
        training_df[col] = training_df[col].replace(values_to_remove, None)
        testing_df[col] = testing_df[col].replace(values_to_remove, None)
    return training_df, testing_df


def replace_numerical_to_categorical(training_df, testing_df):
    logging.info(
        u"Converting DFs to categorical columns, shapes: %s, %s",
        training_df.shape, testing_df.shape
    )

    training_df, testing_df = _clean_numeric_outliers(training_df, testing_df)
    training_df, testing_df = _numeric_to_categorical(training_df, testing_df)
    training_df, testing_df = _clean_categorical_outliers(training_df, testing_df)

    logging.info(
        u"Converted DFs shape: %s, %s",
        training_df.shape, testing_df.shape
    )

    return training_df, testing_df


def main():
    logging.info(u"Start")
    df = pd.read_csv(args.data_csv_path)
    df = canonize_datetime(df, DATE_COLS)
    original_columns = df.columns

    # get training and testing
    tw_splitter = TimeWindowSplitter(args.data_percentage, args.test_percentage)
    training_rows, testing_rows = tw_splitter.split(df.bookdate, args.random_state)
    training_df, testing_df = df.loc[training_rows], df.loc[testing_rows]
    training_df = remove_unrepresentative_users(training_df, args.min_bookings_per_user)
    testing_df = actualize_testing_data(training_df, testing_df)

    # drop & transform cols
    training_df = prepare_for_categorization(training_df)
    testing_df = prepare_for_categorization(testing_df)
    training_df, testing_df = replace_numerical_to_categorical(training_df, testing_df)

    # quality check
    training_columns = set(training_df.columns).union(COLS_TO_DROP + DATE_COLS).difference(['n_booked_days'])
    check_processed_columns(training_columns, original_columns)
    testing_columns = set(training_df.columns).union(COLS_TO_DROP + DATE_COLS).difference(['n_booked_days'])
    check_processed_columns(testing_columns, original_columns)

    training_df.to_csv(args.training_csv, index=False)
    testing_df.to_csv(args.testing_csv, index=False)
    logging.info(u"Finish")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-p", default=1.0, dest="data_percentage", type=float,
                        help=u"Percentage of the data used for testing and training. Default: 1.0")
    parser.add_argument("-t", default=0.2, dest="test_percentage", type=float,
                        help=u"Percentage of the data used for testing purposes. Default: 0.2")
    parser.add_argument("-r", dest="random_state", type=int, required=False,
                        help=u"Random state")
    parser.add_argument("-d", required=True, dest="data_csv_path",
                        help=u"Path to a csv file with the cleaned bookings")
    parser.add_argument("-m", dest="min_bookings_per_user", type=int, default=1,
                        help=u"Min bookings per user. Default: 1")
    parser.add_argument("--trf", default='training.csv', dest="training_csv",
                        help=u"Training data file name. Default: training.csv")
    parser.add_argument("--tsf", default='testing.csv', dest="testing_csv",
                        help=u"Testing data file name. Default: testing.csv")
    parser.add_argument("--log-level", default='INFO', dest="log_level",
                        choices=['DEBUG', 'INFO', 'WARNINGS', 'ERROR'], help=u"Logging level")

    args = parser.parse_args()
    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s', stream=sys.stdout, level=getattr(logging, args.log_level)
    )

    main()
