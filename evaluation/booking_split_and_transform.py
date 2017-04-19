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

from preprocessing.common import canonize_datetime
from feature_matrix.functions import density_based_cutter, fix_outliers
from misc.splitter import TimeWindowSplitter

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


def _simple_cleaning(df):
    df = df.drop(COLS_TO_DROP, axis=1)
    df[u'n_booked_days'] = (df.fdate - df.sdate).apply(lambda x: x.days)
    df = df.drop([u'sdate', u'fdate'], axis=1)
    df.drivetime = np.round(df.drivetime / 3600)  # to hours
    df.n_booked_days = df.n_booked_days.apply(lambda x: 1 if pd.isnull(x) or x < 1 else x)
    df.avg_spend_per_head /= df.n_booked_days.astype(float)
    return df


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
    training_df, testing_df = _convert_to_cat(training_df, testing_df, "adults", 4)
    training_df, testing_df = _convert_to_cat(training_df, testing_df, "children", 3)
    training_df, testing_df = _convert_to_cat(training_df, testing_df, "babies", 3)
    training_df, testing_df = _convert_to_cat(training_df, testing_df, "avg_spend_per_head", 5)
    training_df, testing_df = _convert_to_cat(training_df, testing_df, "n_booked_days", 5)
    training_df, testing_df = _convert_to_cat(training_df, testing_df, "drivetime", 4)
    return training_df, testing_df


def _clean_categorical_outliers(training_df, testing_df, min_p=0.005, max_p=0.995):
    for col in training_df.columns[training_df.dtypes == 'object'].drop(RESERVED_COLS, 'ignore'):
        val_count = training_df[col].dropna().value_counts() / float(training_df.shape[0])
        values_to_remove = val_count[(val_count < min_p) | (val_count > max_p)].index
        training_df[col] = training_df[col].replace(values_to_remove, None)
        testing_df[col] = testing_df[col].replace(values_to_remove, None)
    return training_df, testing_df


def actualize_training_data(training_df, min_bookings_per_user):
    if min_bookings_per_user > 1:
        logging.info(u"Training data, before cleaning: %s", training_df.shape)
        bookings_per_user = training_df.code.value_counts()
        good_user_ids = bookings_per_user[bookings_per_user >= min_bookings_per_user].index
        training_df = training_df[training_df.code.isin(good_user_ids)]
        logging.info(u"Training data, after cleaning: %s", training_df.shape)
    return training_df


def main():
    logging.info(u"Start")
    df = pd.read_csv(args.data_csv_path)
    df = canonize_datetime(df, DATE_COLS)

    # get training and testing
    tw_splitter = TimeWindowSplitter(args.data_percentage, args.test_percentage)
    training_rows, testing_rows = tw_splitter.split(df.bookdate, args.random_state)
    training_df, testing_df = df.loc[training_rows], df.loc[testing_rows]
    training_df = actualize_training_data(training_df, args.min_bookings_per_user)
    testing_df = actualize_testing_data(training_df, testing_df)

    # drop & transform cols
    training_df = _simple_cleaning(training_df)
    testing_df = _simple_cleaning(testing_df)

    # fix outliers
    training_df, testing_df = _clean_numeric_outliers(training_df, testing_df)

    # convert numeric to categorical
    training_df, testing_df = _numeric_to_categorical(training_df, testing_df)

    # clean categorical outliers
    training_df, testing_df = _clean_categorical_outliers(training_df, testing_df)
    logging.info(u"Training shape: %s", training_df.shape)
    logging.info(u"Testing shape: %s", testing_df.shape)

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
