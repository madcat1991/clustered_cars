"""
This script cleans and prepares the data set of properties for the future usage
"""

import argparse
import logging
import sys

import pandas as pd

from preprocessing.common import raw_data_to_df, check_processed_columns, check_data

COLS_TO_DROP = [
    u'callflag', u'placeid',  # no need
    u'createdate', u'changeday',  u'bandno', # no need
    u'pageno',  u'gridx', u'gridy',  # no need
    u'commission', u'latitude', u'longitude',  # no need
    u'bandchar', u'opinion',  # no need
]

INT_COLS = [
    u'active', u'bathrooms', u'bedrooms', u'bunks',
    u'doubles', u'familyrooms', u'sleeps', u'twin',
]

BOOL_COLS = [
    u'shortbreakok'
]

NOT_NA_COLS = [u'propcode', u'propid', u'year', u'active']


def main():
    check_data(args.input_csv, args.input_csv_delimiter)
    df = raw_data_to_df(args.input_csv, args.input_csv_delimiter)
    original_columns = df.columns
    logging.info(u"DF initial shape: %s", df.shape)

    logging.info(u"Cleaning data")
    df = df.drop(COLS_TO_DROP, axis=1)
    df = df.dropna(subset=NOT_NA_COLS)
    logging.info(u"Shape after cleaning: %s", df.shape)

    # df[BOOL_COLS] = df[BOOL_COLS].apply(lambda x: x.str.contains('true', case=False, na=False)).astype(int)
    df[BOOL_COLS] = df[BOOL_COLS].astype(int)
    df[INT_COLS] = df[INT_COLS].apply(lambda x: pd.to_numeric(x))
    df.stars = pd.to_numeric(df.stars, errors='coerce')

    processed_columns = set(df.columns).union(COLS_TO_DROP)
    check_processed_columns(processed_columns, original_columns)

    logging.info(u"Dumping data to: %s", args.output_csv)
    df.to_csv(args.output_csv, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', required=True, dest="input_csv",
                        help=u'Path to a csv file with properties')
    parser.add_argument('--id', default=";", dest="input_csv_delimiter",
                        help=u"The input file's delimiter. Default: ';'")
    parser.add_argument('-o', default="properties.csv", dest="output_csv",
                        help=u'Path to an output file. Default: property.csv')
    parser.add_argument("--log-level", default='INFO', dest="log_level",
                        choices=['DEBUG', 'INFO', 'WARNINGS', 'ERROR'], help=u"Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s', stream=sys.stdout, level=getattr(logging, args.log_level)
    )

    main()
