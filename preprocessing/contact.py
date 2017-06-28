"""
This script cleans and prepares the data set of contacts for the future usage
"""

import argparse
import logging
import sys

from preprocessing.common import raw_data_to_df, check_processed_columns, check_data

COLS_TO_DROP = [
    u'createdate', u'last_brochure_date',  # no need
    u'outcode', u'donotemail', u'total_hh_net', u'total_ho',  # no need
    u'oac_supergroup', u'oac_group', u'oac_subgroup',  # took the same with *_desc
    u'last_bookdate', u'last_book_year',  # can be taken from bookings
    u'first_bookdate', u'first_book_year', u'last_sdate', u'first_sdate',  # can be taken from bookings
    u'last_brochure_year', u'available', u'decile',  # no need
    u'mosaic_group', u'mosaic_group_desc', u'mosaic_type', u'mosaic_type_desc',  # redundant
    u'recency',  # no need
    u'origsourcecostid',  # is a pair of u'origsourcedesc', u'origsourcecategory'
    u'currentsourcecostid', u'currentsourcedesc', u'currentsourcecategory', u'ff'  # no need
]

NOT_NA_COLS = [u"code"]


def main():
    check_data(args.input_csv, args.input_csv_delimiter)
    df = raw_data_to_df(args.input_csv, args.input_csv_delimiter)
    original_columns = df.columns
    logging.info(u"DF initial shape: %s", df.shape)

    logging.info(u"Cleaning data")
    df = df.drop(COLS_TO_DROP, axis=1)
    df = df.dropna(subset=NOT_NA_COLS)
    logging.info(u"Shape after cleaning: %s", df.shape)

    processed_columns = set(df.columns).union(COLS_TO_DROP)
    check_processed_columns(processed_columns, original_columns)

    logging.info(u"Dumping data to: %s", args.output_csv)
    df.to_csv(args.output_csv, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', required=True, dest="input_csv",
                        help=u'Path to a csv file with contacts')
    parser.add_argument('--id', default=";", dest="input_csv_delimiter",
                        help=u"The input file's delimiter. Default: ';'")
    parser.add_argument('-o', default="HH_Cleaned_Contact.csv", dest="output_csv",
                        help=u'Path to an output file. Default: contact.csv')
    parser.add_argument("--log-level", default='INFO', dest="log_level",
                        choices=['DEBUG', 'INFO', 'WARNINGS', 'ERROR'], help=u"Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s', stream=sys.stdout, level=getattr(logging, args.log_level)
    )

    main()
