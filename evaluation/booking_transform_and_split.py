"""
The script splits a booking csv dataset into testing and training
parts based on timestamp column. Then it transforms the parts
into binary mode.
"""

import argparse
import logging
import sys

import pandas as pd

from model.booking_transform import remove_unrepresentative_users, prepare_for_categorization, \
    DATE_COLS, COLS_TO_DROP, BINNING_COLS
from preprocessing.common import canonize_datetime, check_processed_columns
from feature_matrix.functions import fix_outliers, get_bins_for_num_column
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


def replace_numerical_to_categorical_train_and_test(training_df, testing_df, min_p=0.005, max_p=0.995):
    logging.info(u"Binning DF numerical columns to categorical: %s", BINNING_COLS)
    for col, n_bins in BINNING_COLS.items():
        logging.info("Binning columns '%s'", col)
        col_data_train, min_val, max_val = fix_outliers(training_df[col].copy(), return_min_max=True)
        col_data_test = fix_outliers(testing_df[col].copy(), min_val=min_val, max_val=max_val)

        bins = get_bins_for_num_column(col_data_train, n_bins)
        col_data_train = pd.cut(col_data_train, bins, include_lowest=True).astype('object')
        col_data_test = pd.cut(col_data_test, bins, include_lowest=True).astype('object')

        val_count = col_data_train.dropna().value_counts() / float(col_data_train.shape[0])
        values_to_remove = val_count[(val_count < min_p) | (val_count > max_p)].index

        training_df[col] = col_data_train.replace(values_to_remove, None)
        testing_df[col] = col_data_test.replace(values_to_remove, None)
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
    training_df, testing_df = replace_numerical_to_categorical_train_and_test(training_df, testing_df)

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
