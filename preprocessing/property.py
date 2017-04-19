"""
This script cleans and prepares the data set of properties for the future usage
"""

import argparse
import logging
import sys

import pandas as pd

COLS_TO_DROP = [
    u'active', u'callflag', u'placeid',  # no need
    u'createdate', u'changeday',  u'bandno', # no need
    u'pageno',  u'gridx', u'gridy',  # no need
    u'commission', u'latitude', u'longitude',  # no need
    # u'bandchar', u'sb_bandno', u'opinion', ## changed to the new names
    u'BandChar', u'Opinion',
]

INT_COLS = [
    u'bathrooms', u'bedrooms', u'bunks', u'doubles',
    u'familyrooms', u'sleeps', u'twin',
    # u'shortbreakok', u'singles'
]

BOOL_COLS = [
    u'shortbreakok'
]

NOT_NA_COLS = [u'propcode', u'propid', u'year']


def main():
    df = pd.read_csv(args.input_csv, delimiter=args.input_csv_delimiter)
    logging.info(u"DF initial shape: %s", df.shape)

    logging.info(u"Cleaning data")
    df = df.drop(COLS_TO_DROP, axis=1)
    df = df.dropna(subset=NOT_NA_COLS)

    df[BOOL_COLS] = df[BOOL_COLS].apply(lambda x: x.str.contains('true', case=False, na=False)).astype(int)
    df[INT_COLS] = df[INT_COLS].apply(lambda x: pd.to_numeric(x))

    logging.info(u"Shape after cleaning: %s", df.shape)

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
