"""
This script cleans and prepares the data set of bookings for the future usage
"""

import argparse
import logging
import sys

import pandas as pd

from preprocessing.common import canonize_datetime, raw_data_to_df, check_processed_columns, check_data

OLD_BREAKPOINT_MATCHER = {
    2001: [
        (1, 1, "New Year"), (1, 6, "Winter"),
        (2, 17, "Half Terms"), (2, 24, "Spring and Autumn"),
        (4, 7, "Easter"), (4, 21, "Spring and Autumn"),
        (5, 26, "SBH"),
        (6, 2, "Early Summer"),
        (7, 21, "Summer holidays"),
        (9, 1, "Early Autumn"), (9, 15, "Spring and Autumn"),
        (10, 27, "Half Terms"),
        (11, 3, "Winter"),
        (12, 22, "Christmas"), (12, 29, "New Year"),
    ],
    2002: [
        (1, 1, "New Year"), (1, 5, "Winter"),
        (2, 16, "Half Terms"), (2, 23, "Spring and Autumn"),
        (4, 6, "Easter"), (4, 20, "Spring and Autumn"),
        (5, 25, "SBH"),
        (6, 1, "Early Summer"),
        (7, 20, "Summer holidays"),
        (8, 31, "Early Autumn"),
        (9, 14, "Spring and Autumn"),
        (10, 26, "Half Terms"),
        (11, 2, "Winter"),
        (12, 21, "Christmas"), (12, 28, "New Year"),
    ],
    2003: [
        (1, 1, "New Year"), (1, 4, "Winter"),
        (2, 15, "Half Terms"), (2, 22, "Spring and Autumn"),
        (4, 5, "Easter"), (4, 19, "Spring and Autumn"),
        (5, 24, "SBH"), (5, 31, "Early Summer"),
        (7, 19, "Summer holidays"),
        (8, 30, "Early Autumn"),
        (9, 13, "Spring and Autumn"),
        (10, 25, "Half Terms"),
        (11, 1, "Winter"),
        (12, 20, "Christmas"), (12, 27, "New Year"),
    ],
    2004: [
        (1, 1, "New Year"), (1, 3, "Winter"),
        (2, 14, "Half Terms"), (2, 21, "Spring and Autumn"),
        (4, 3, "Easter"), (4, 17, "Spring and Autumn"),
        (5, 22, "SBH"), (5, 29, "Early Summer"),
        (7, 17, "Summer holidays"),
        (8, 28, "Early Autumn"),
        (9, 11, "Spring and Autumn"),
        (10, 23, "Half Terms"), (10, 30, "Winter"),
        (12, 18, "Christmas"),
    ],
    2005: [
        (1, 1, "Winter"),
        (2, 12, "Half Terms"), (2, 19, "Spring and Autumn"),
        (4, 2, "Easter"), (4, 16, "Spring and Autumn"),
        (5, 21, "SBH"), (5, 28, "Early Summer"),
        (7, 16, "Summer holidays"),
        (8, 27, "Early Autumn"),
        (9, 10, "Spring and Autumn"),
        (10, 22, "Half Terms"), (10, 29, "Winter"),
        (12, 17, "Christmas"), (12, 31, "New Year"),
    ],
    2006: [
        (1, 1, "New Year"), (1, 7, "Winter"),
        (2, 18, "Half Terms"), (2, 25, "Spring and Autumn"),
        (4, 8, "Easter"), (4, 22, "Spring and Autumn"),
        (5, 27, "SBH"),
        (6, 3, "Early Summer"),
        (7, 22, "Summer holidays"),
        (9, 2, "Early Autumn"), (9, 16, "Spring and Autumn"),
        (10, 28, "Half Terms"),
        (11, 4, "Winter"),
        (12, 23, "Christmas"), (12, 30, "New Year"),
    ],
    2007: [
        (1, 1, "New Year"), (1, 6, "Winter"),
        (2, 17, "Half Terms"), (2, 24, "Spring and Autumn"),
        (4, 7, "Easter"),
        (4, 21, "Spring and Autumn"),
        (5, 26, "SBH"),
        (6, 2, "Early Summer"),
        (7, 21, "Summer holidays"),
        (9, 1, "Early Autumn"), (9, 15, "Spring and Autumn"),
        (10, 27, "Half Terms"),
        (11, 3, "Winter"),
        (12, 22, "Christmas"), (12, 29, "New Year"),
    ],
    2008: [
        (1, 1, "New Year"), (1, 5, "Winter"),
        (2, 16, "Half Terms"), (2, 23, "Spring and Autumn"),
        (3, 22, "Easter"),
        (4, 19, "Spring and Autumn"),
        (5, 24, "SBH"), (5, 31, "Early Summer"),
        (7, 19, "Summer holidays"),
        (8, 30, "Early Autumn"),
        (9, 13, "Spring and Autumn"),
        (10, 25, "Half Terms"),
        (11, 1, "Winter"),
        (12, 20, "Christmas"),
    ],
}

COLS_TO_DROP = [
    'pname', 'region', 'sleeps', 'stars', 'proppostcode',  # can be taken from property
    'bookdate_scoreboard', 'book_year', 'hh_gross', 'hh_net', 'ho',  # HH specific
    'holidayprice',  # correlates with avg_spend_per_head
    'bighouse', 'burghisland', 'boveycastle',  # no need
    'sourcecostid',  # is a pair of u'sourcedesc', u'category'
    'drivedistance',  # correlates with drivetime
]

NOT_NA_COLS = [u'bookcode', u'code', u'propcode', u'year', u'breakpoint', u'avg_spend_per_head']
DATE_COLS = [u'bookdate', u'sdate', u"fdate"]
FLOAT_COLS = [u'avg_spend_per_head', u'drivetime']
INT_COLS = [u'adults', u'babies', u'children', u'pets']
CATEGORICAL_COLS = [u'sourcedesc', u'category']


def get_breakpoint(dt):
    breakpoint = None
    matcher = OLD_BREAKPOINT_MATCHER.get(dt.year, [])
    for _m, _d, _b in matcher:
        if _m > dt.month or (_m == dt.month and _d > dt.day):
            break
        breakpoint = _b
    return breakpoint


def fine_tune_df(df):
    logging.info(u"DF shape before fine tuning: %s", df.shape)

    averages = {col: df[col].dropna().mean() for col in FLOAT_COLS}
    zeros = {col: 0 for col in INT_COLS}
    most_popular_values = {col: df[col].value_counts().index[0] for col in CATEGORICAL_COLS}

    logging.info(u"Filling NA with average: %s", averages)
    df = df.fillna(averages)
    logging.info(u"Filling NA with zeros: %s", zeros)
    df = df.fillna(zeros)
    logging.info(u"Filling NA with most populars: %s", most_popular_values)
    df = df.fillna(most_popular_values)

    df[INT_COLS] = df[INT_COLS].astype(int)

    logging.info(u"Before cleaning NA: %s", df.shape)
    df = df.dropna(subset=NOT_NA_COLS)
    logging.info(u"After cleaning NA: %s", df.shape)

    if pd.isnull(df.values).any():
        logging.error(u"NA values left in df")
    return df


def fill_missed_breakpoints(df):
    df = df[pd.notnull(df.breakpoint) | pd.notnull(df.zone_name)]
    logging.info(u"Bookings having breakpoint or zone_name: %s", df.shape[0])
    logging.info(u"Filling missing breakpoints: %s", df[pd.isnull(df.breakpoint)].shape[0])
    df.breakpoint[pd.isnull(df.breakpoint)] = df[pd.isnull(df.breakpoint)].sdate.apply(get_breakpoint)
    logging.info(u"Left NA breakpoints: %s", df[pd.isnull(df.breakpoint)].shape[0])
    return df.drop(u'zone_name', axis=1)


def main():
    check_data(args.input_csv, args.input_csv_delimiter)
    df = raw_data_to_df(args.input_csv, args.input_csv_delimiter)
    original_columns = df.columns
    logging.info(u"DF initial shape: %s", df.shape)

    df = df.drop(COLS_TO_DROP, axis=1)

    df = canonize_datetime(df, DATE_COLS)

    df = fill_missed_breakpoints(df)
    df = fine_tune_df(df)

    processed_columns = set(df.columns).union(COLS_TO_DROP + [u'zone_name'])
    check_processed_columns(processed_columns, original_columns)

    logging.info(u"Dumping data to: %s", args.output_csv)
    df.to_csv(args.output_csv, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', required=True, dest="input_csv",
                        help=u'Path to a csv file with bookings')
    parser.add_argument('--id', default=";", dest="input_csv_delimiter",
                        help=u"The input file's delimiter. Default: ';'")
    parser.add_argument('-o', default="bookings.csv", dest="output_csv",
                        help=u'Path to an output file. Default: booking.csv')
    parser.add_argument("--log-level", default='INFO', dest="log_level",
                        choices=['DEBUG', 'INFO', 'WARNINGS', 'ERROR'], help=u"Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s', stream=sys.stdout, level=getattr(logging, args.log_level)
    )

    main()
