# coding: utf-8

u"""
This script converts booking training and testing data sets into
CARSKit style representation
"""


import argparse
import logging

import pandas as pd
import sys


def main():
    logging.info(u"Start")
    training_df = pd.read_csv(args.training_csv)
    testing_df = pd.read_csv(args.testing_csv)

    tr_keys = training_df[["code", "propcode"]].drop_duplicates()
    t_keys = testing_df[["code", "propcode"]].drop_duplicates()
    assert pd.merge(tr_keys, t_keys, on=["code", "propcode"]).shape[0] == 0, \
        "Training and testing data has the same user-item pairs. Not good!"

    # CARSKit needs rating column
    training_df["rating"] = 1
    testing_df["rating"] = 1

    rename_d = {"code": "user", "propcode": "item"}
    reserved_cols = ["user", "item", "rating"]
    other_cols = [
        'breakpoint', 'adults', 'children', 'babies', 'avg_spend_per_head',
        'pets', 'category', 'drivetime', 'booking_days'
    ]

    # we removed bookcode and year, it's time to remove duplicates
    logging.info("Initial shapes: %s, %s", training_df.shape, testing_df.shape)
    training_df = training_df.rename(columns=rename_d)[reserved_cols + other_cols]
    training_df = training_df.drop_duplicates()

    testing_df = testing_df.rename(columns=rename_d)[reserved_cols + other_cols]
    testing_df = testing_df.drop_duplicates()
    logging.info("Shapes after drop_duplicates: %s, %s", training_df.shape, testing_df.shape)

    df = pd.concat([training_df, testing_df])

    # CARSKit's founders are not able use already existing csv parsing tools and wrote their own one
    # let's remove any unexpected symbol from the context values
    # simple hashing
    value_index = {}
    logging.info(u"Hashing data")
    for col in df.columns[training_df.dtypes == 'object'].drop(reserved_cols, 'ignore'):
        df[col] = df[col].apply(lambda x: value_index.setdefault(x, len(value_index)) if pd.notnull(x) else None)

    print "Replacement table:"
    for k, v in sorted(value_index.items(), key=lambda x: x[1]):
        print "-> %s. :%s" % (v, k)

    # CARSKit way of handling null values
    df = df.fillna('NA')

    logging.info(u"Dumping data")
    df.to_csv(args.ck_data_csv, index=False)
    logging.info(u"Finish")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--trf", default='training.csv', dest="training_csv",
                        help=u"Training data file name. Default: training.csv")
    parser.add_argument("--tsf", default='testing.csv', dest="testing_csv",
                        help=u"Testing data file name. Default: testing.csv")
    parser.add_argument("--ctr", default='ck_data.csv', dest="ck_data_csv",
                        help=u"CARSKit data file name. Default: ck_data.csv")
    parser.add_argument("--log-level", default='INFO', dest="log_level",
                        choices=['DEBUG', 'INFO', 'WARNINGS', 'ERROR'], help=u"Logging level")

    args = parser.parse_args()
    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s', stream=sys.stdout, level=getattr(logging, args.log_level)
    )

    main()
