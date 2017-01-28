# coding: utf-8

u"""
The script prepares item-feature vectors
"""

import logging
import sys

import numpy as np
import pandas as pd

from feature_matrix.functions import prepare_num_column


def get_bdf():
    bdf = pd.read_csv(args.booking_csv)
    booking_cols = [
        'bookcode', 'year', 'propcode',
        'breakpoint', 'adults', 'children', 'babies',
        'avg_spend_per_head', 'pets'
    ]
    return bdf[booking_cols]


def get_idf(bdf):
    idf = pd.merge(
        pd.read_csv(args.property_csv),
        bdf[["propcode", "year"]].drop_duplicates(),
        on=["propcode", "year"]
    )
    item_cols = ['propcode', 'year', 'region', 'stars']
    idf = idf[item_cols]
    idf.stars = prepare_num_column(idf.stars)
    return idf


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
        'broadband',
        'coast 5 miles',
        'complex',
        'countryside views',
        'detached',
        'enclosed garden',
        'enhanced',
        'farm help',
        'fishing - private',
        'freezer',
        'fridge',
        'fridge-freezer',
        'games room',
        'golf course nearby - good',
        'good for honeymooners',
        'hot tub',
        'indoor pool',
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
        'travel cot',
        'tumble drier',
        'video',
        'vineyard',
        'washer-drier',
        'washing machine',
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

    # only features related to items and years from bdf
    fdf = get_fdf(bdf)

    # preparing item vectors
    df = pd.merge(bdf, idf, on=["propcode", "year"], how='left')
    df = pd.merge(df, fdf, on=["propcode", "year"], how='left')
    df = df.drop(["year"], axis=1)

    cols_to_binarize = df.columns[df.dtypes == 'object'].drop(["bookcode", "propcode"])
    df = pd.get_dummies(df, columns=cols_to_binarize).fillna(0)

    # dropping columns that can't change anything
    feature_cols = df.columns.drop(["bookcode", "propcode"])
    bad_col_ids = np.where(df[feature_cols].sum(axis=0) < 50)[0]
    df = df.drop(feature_cols[bad_col_ids], axis=1)
    logging.info(u"Dumping prepared booking-feature matrix: %s", df.shape)
    df.to_csv(args.output_csv, index=False)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    # parser.add_argument("-b", required=True, dest="booking_csv", help=u"Path to a csv file with bookings")
    # parser.add_argument("-c", required=True, dest="contact_csv", help=u"Path to a csv file with contacts")
    # parser.add_argument("-p", required=True, dest="property_csv", help=u"Path to a csv file with properties")
    # parser.add_argument("-f", required=True, dest="feature_csv", help=u"Path to a csv file with features")
    # parser.add_argument('-o', default="user_features.csv", dest="output_csv",
    #                     help=u'Path to an output file. Default: user_features.csv')
    # parser.add_argument("--log-level", default='INFO', dest="log_level",
    #                     choices=['DEBUG', 'INFO', 'WARNINGS', 'ERROR'], help=u"Logging level")
    #
    # args = parser.parse_args()

    from collections import namedtuple

    args = namedtuple(
        "args",
        ["booking_csv", "contact_csv", "property_csv", "feature_csv", "log_level", "output_csv"]
    )

    args.booking_csv = '/Users/user/PyProjects/clustered_cars/data/training.csv'
    args.contact_csv = '/Users/user/PyProjects/clustered_cars/data/HH_Cleaned_Contact.csv'
    args.property_csv = '/Users/user/PyProjects/clustered_cars/data/HH_Cleaned_Property.csv'
    args.feature_csv = '/Users/user/PyProjects/clustered_cars/data/HH_Cleaned_Features.csv'
    args.output_csv = '/Users/user/PyProjects/clustered_cars/data/item_features.csv'
    args.log_level = 'INFO'

    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s', stream=sys.stdout, level=getattr(logging, args.log_level)
    )

    main()
