# coding: utf-8

u"""
The script prepares user-feature vectors
"""
import argparse
import logging
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import binarize

from feature_matrix.functions import prepare_num_column


def get_bdf():
    bdf = pd.read_csv(args.booking_csv)
    booking_cols = [
        "code", "year", "breakpoint", "propcode",
        "pets", "category", "drivetime", "booking_days",
        "avg_spend_per_head"
    ]
    return bdf[booking_cols]


def get_idf(bdf):
    idf = pd.merge(
        pd.read_csv(args.property_csv),
        bdf[["propcode", "year"]].drop_duplicates(),
        on=["propcode", "year"]
    )
    item_cols = ["propcode", "year", "stars"]
    idf = idf[item_cols]
    idf.stars = prepare_num_column(idf.stars)
    return idf


def get_udf(bdf):
    udf = pd.read_csv(args.contact_csv)
    user_cols = ["code", "oac_groupdesc"]
    udf = udf[udf.code.isin(bdf.code)][user_cols]
    return udf


def get_fdf(bdf):
    fdf = pd.merge(
        pd.read_csv(args.feature_csv),
        bdf[["propcode", "year"]].drop_duplicates(),
        on=["propcode", "year"]
    )
    feature_cols = [
        'baby sitting',
        'barbecue',
        'big gardens or farm to wander',
        'boats or mooring available',
        'coast 5 miles',
        'complex',
        'countryside views',
        'detached',
        'enclosed garden',
        'enhanced',
        'farm help',
        'fishing - private',
        'games room',
        'golf course nearby - good',
        'hot tub',
        'on a farm',
        'open fire or woodburner',
        'outdoor heated pool',
        'part disabled',
        'piano',
        'pool',
        'pub 1 mile walk',
        'railway 5 miles',
        'sailing nearby',
        'sandy beach 1 mile',
        'sauna',
        'sea views',
        'shooting',
        'snooker table',
        'stairgate',
        'tennis court',
        'vineyard',
        'wheel chair facilities'
    ]
    fdf = fdf[["propcode", "year"] + feature_cols]
    # converting to binary
    fdf.enhanced = fdf.enhanced.apply(lambda x: 0 if pd.isnull(x) else 1)
    fdf.vineyard = fdf.vineyard.apply(lambda x: 0 if pd.isnull(x) else 1)
    fdf.stairgate = fdf.stairgate.apply(lambda x: 0 if pd.isnull(x) or x == 'no' else 1)
    fdf[feature_cols] = fdf.fillna(0)[feature_cols].astype(bool).astype(int)
    return fdf


def main():
    # we start from bookings, since it is a training part
    bdf = get_bdf()

    # only items that have been seen in bookings
    idf = get_idf(bdf)

    # only users that have been seen in bookings
    udf = get_udf(bdf)

    # only features related to items and years from bdf
    fdf = get_fdf(bdf)

    # preparing user vectors
    df = pd.merge(bdf, udf, on=["code"], how='left')
    df = pd.merge(df, idf, on=["propcode", "year"], how='left')
    df = pd.merge(df, fdf, on=["propcode", "year"], how='left')
    df = df.drop(["propcode", "year"], axis=1)

    cols_to_binarize = df.columns[df.dtypes == 'object'].drop("code")
    df = pd.get_dummies(df, columns=cols_to_binarize).fillna(0)
    df["booking_cnt"] = 1  # for future purposes
    df = df.groupby("code").sum().reset_index()

    # dropping columns that can't change anything
    feature_cols = df.columns.drop("code")
    bad_col_ids = np.where(binarize(df[feature_cols]).sum(axis=0) < args.min_values_per_column)[0]
    df = df.drop(feature_cols[bad_col_ids], axis=1)
    logging.info(u"Dumping prepared user-feature matrix: %s", df.shape)
    df.to_csv(args.output_csv, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-b", required=True, dest="booking_csv", help=u"Path to a csv file with bookings")
    parser.add_argument("-c", required=True, dest="contact_csv", help=u"Path to a csv file with contacts")
    parser.add_argument("-p", required=True, dest="property_csv", help=u"Path to a csv file with properties")
    parser.add_argument("-f", required=True, dest="feature_csv", help=u"Path to a csv file with features")
    parser.add_argument('-o', default="user_features.csv", dest="output_csv",
                        help=u'Path to an output file. Default: user_features.csv')
    parser.add_argument('-m', default=2, type=int, dest="min_values_per_column",
                        help=u'Min binary values per column. Default: 2')
    parser.add_argument("--log-level", default='INFO', dest="log_level",
                        choices=['DEBUG', 'INFO', 'WARNINGS', 'ERROR'], help=u"Logging level")

    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s', stream=sys.stdout, level=getattr(logging, args.log_level)
    )

    main()
