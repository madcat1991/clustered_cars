"""
The script prepares booking-feature vectors
"""
import argparse
import logging
import sys

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
    item_cols = ['propcode', 'year', 'region', 'stars', 'sleeps']
    idf = idf[item_cols]
    idf.stars = prepare_num_column(idf.stars)
    idf.sleeps = prepare_num_column(idf.sleeps, bins=5)
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
        'games room',
        'golf course nearby - good',
        'good for honeymooners',
        'high chair',
        'hot tub',
        'indoor pool',
        'jacuzzi',
        'on a farm',
        'open fire or woodburner',
        'outdoor heated pool',
        'parking',
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
        'tennis court',
        'travel cot',
        'video',
        'vineyard',
        'wheel chair facilities'
    ]
    fdf = fdf[["propcode", "year"] + feature_cols]
    fdf.enhanced = fdf.enhanced.apply(lambda x: 0 if pd.isnull(x) else 1)
    fdf.vineyard = fdf.vineyard.apply(lambda x: 0 if pd.isnull(x) else 1)
    fdf.parking = fdf.parking.apply(lambda x: 0 if pd.isnull(x) else 1)
    # converting to binary
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
    logging.info("Shape before cleaning: %s", df.shape)

    # dropping columns that can't change anything
    bad_feature_cols = []
    for feature_col in df.columns.drop(["bookcode", "propcode"]):
        items_per_feature = df.propcode[df[feature_col] == 1].unique().size
        if items_per_feature < args.min_items_per_feature:
            bad_feature_cols.append(feature_col)

    logging.info("Bad columns: %s", bad_feature_cols)
    df = df.drop(bad_feature_cols, axis=1)

    logging.info(u"Dumping prepared booking-feature matrix: %s", df.shape)
    df.to_csv(args.output_csv, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-b", required=True, dest="booking_csv", help=u"Path to a csv file with bookings")
    parser.add_argument("-p", required=True, dest="property_csv", help=u"Path to a csv file with properties")
    parser.add_argument("-f", required=True, dest="feature_csv", help=u"Path to a csv file with features")
    parser.add_argument('-o', default="bookings.csv", dest="output_csv",
                        help=u'Path to an output file. Default: bookings.csv')
    parser.add_argument('-m', default=10, type=int, dest="min_items_per_feature",
                        help=u'Min items per feature. Default: 10')
    parser.add_argument("--log-level", default='INFO', dest="log_level",
                        choices=['DEBUG', 'INFO', 'WARNINGS', 'ERROR'], help=u"Logging level")

    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s', stream=sys.stdout, level=getattr(logging, args.log_level)
    )

    main()
