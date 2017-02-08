# coding: utf-8

u"""
The script transforms all bookings into binary mode.
"""

import argparse
import logging

import pandas as pd
import sys

from feature_matrix.functions import prepare_num_column


COLS_TO_BIN = ["adults", "children", "babies", "avg_spend_per_head", "booking_days", "drivetime"]
RESERVED_COLS = ["code", "year", "propcode", "bookcode"]
COLS_TO_DROP = [
    u'bookdate',  # no need
    u'drivedistance',  #correlatets with drivetime
    u'holidayprice',  # correlates with avg_spend_per_head and number of users
    u'burghisland', u'boveycastle', u'bighouse',  # no need
    u'sourcedesc'  # too detailed
]


def _simple_cleaning(df):
    df = df.drop(COLS_TO_DROP, axis=1)
    df[u'booking_days'] = \
        (pd.to_datetime(df.fdate, dayfirst=True) - pd.to_datetime(df.sdate, dayfirst=True)).apply(lambda x: x.days)
    df = df.drop([u'sdate', u'fdate'], axis=1)
    df.drivetime /= 3600  # to hours
    df.booking_days = df.booking_days.apply(lambda x: 1 if pd.isnull(x) or x < 1 else x)
    df.avg_spend_per_head /= df.booking_days.astype(float)
    return df


def _clean_categorical_outliers(df, min_p=0.005):
    for col in df.columns[df.dtypes == 'object'].drop(RESERVED_COLS, 'ignore'):
        val_p = df[col].dropna().value_counts() / float(df.shape[0])
        values_to_remove = val_p[val_p < min_p].index
        df[col] = df[col].replace(values_to_remove, None)
    return df


def actualize_data(df, min_bookings_per_user):
    logging.info(u"Data shape before actualization: %s", df.shape)
    bookings_per_user = df.code.value_counts()
    good_user_ids = bookings_per_user[bookings_per_user >= min_bookings_per_user].index

    df = df[df.code.isin(good_user_ids)]
    logging.info(u"Data shape after actualization: %s", df.shape)
    return df


def main():
    logging.info(u"Start")
    df = pd.read_csv(args.data_csv_path)
    df = actualize_data(df, args.min_bookings_per_user)

    # drop & transform cols
    df = _simple_cleaning(df)

    # fix outliers & convert numeric to categorical
    df.adults = prepare_num_column(df.adults, min_bin_p=0.05, min_value=1).astype('object')
    df.children = prepare_num_column(df.children, min_bin_p=0.03, min_value=1).astype('object')
    df.babies = prepare_num_column(df.babies, min_bin_p=0.001, min_value=1).astype('object')
    df.avg_spend_per_head = prepare_num_column(df.avg_spend_per_head, min_bin_p=0.06, min_value=1).astype('object')
    df.booking_days = prepare_num_column(df.booking_days, min_bin_p=0.01, min_value=1).astype('object')
    df.drivetime = prepare_num_column(df.drivetime, min_bin_p=0.05).astype('object')

    # clean categorical outliers
    df = _clean_categorical_outliers(df)
    logging.info(u"Data shape: %s", df.shape)

    df.to_csv(args.output, index=False)
    logging.info(u"Finish")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-d", required=True, dest="data_csv_path",
                        help=u"Path to a booking csv dataset")
    parser.add_argument("-o", default='model_bookings.csv', dest="output",
                        help=u"Output data file name. Default: model_bookings.csv")
    parser.add_argument("-m", dest="min_bookings_per_user", type=int, default=3,
                        help=u"Min bookings per user")
    parser.add_argument("--log-level", default='INFO', dest="log_level",
                        choices=['DEBUG', 'INFO', 'WARNINGS', 'ERROR'], help=u"Logging level")

    args = parser.parse_args()
    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s', stream=sys.stdout, level=getattr(logging, args.log_level)
    )

    main()
