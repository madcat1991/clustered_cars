"""
The script prepares item-feature vectors
"""
import argparse
import logging
import sys

import pandas as pd

from model.booking_transform import replace_numerical_to_categorical

RESERVED_COLS = ["propcode", "year"]
BINNING_COLS = {
    'stars': 4,
    'sleeps': 5,
}


def get_idf():
    idf = pd.read_csv(args.property_csv)
    item_cols = ['propcode', 'year', 'region', 'stars', 'sleeps', 'shortbreakok']
    logging.info("Skipped property columns: %s", set(idf.columns).difference(item_cols))
    idf = idf[item_cols]
    idf = replace_numerical_to_categorical(idf, BINNING_COLS)
    return idf


def get_fdf(idf):
    fdf = pd.merge(
        pd.read_csv(args.feature_csv),
        idf[["propcode", "year"]].drop_duplicates(),
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
        'wheel chair facilities',
        'safety deposit box',
        'river or estuary views',
        'outdoor unheated pool',
        'no smoking',
        'steam room',
        'wedding venue',
        'maid service',
        'pamper by the pool',
        'childrens play area',
        'extra infant equipment',
        'views',
        'air conditioning',
        'cycle hire available',
    ]
    logging.info("Skipped features: %s", set(fdf.columns).difference(feature_cols))
    fdf = fdf[["propcode", "year"] + feature_cols]
    # converting to binary
    fdf.enhanced = fdf.enhanced.apply(lambda x: 0 if pd.isnull(x) else 1)
    fdf.vineyard = fdf.vineyard.apply(lambda x: 0 if pd.isnull(x) else 1)
    fdf.parking = fdf.parking.apply(lambda x: 0 if pd.isnull(x) else 1)
    fdf[feature_cols] = fdf.fillna(0)[feature_cols].astype(bool).astype(int)
    return fdf


def main():
    # all items that we have
    idf = get_idf()

    # only features related to items and years from bdf
    fdf = get_fdf(idf)

    # preparing item vectors
    df = pd.merge(idf, fdf, on=["propcode", "year"], how='left')

    cols_to_binarize = df.columns[df.dtypes == 'object'].drop(RESERVED_COLS, errors='ignore')
    df = pd.get_dummies(df, columns=cols_to_binarize).fillna(0)
    logging.info("Shape before cleaning: %s", df.shape)

    # dropping columns that can't change anything
    bad_feature_cols = []
    for feature_col in df.columns.drop(RESERVED_COLS):
        items_per_feature = df.propcode[df[feature_col] == 1].unique().size
        if items_per_feature < args.min_items_per_feature:
            bad_feature_cols.append(feature_col)

    logging.info("Features with less than %s items: %s", args.min_items_per_feature, bad_feature_cols)
    df = df.drop(bad_feature_cols, axis=1)

    # dropping columns that don't present in the last year and in less than 50% of the years
    bad_feature_cols = []
    n_years = df.year.unique().size
    max_year = df.year.unique().max()
    for col in df.columns.drop(RESERVED_COLS):
        col_years = df[df[col] > 0].year.unique()
        if max(col_years) != max_year or len(col_years) < n_years * 0.5:
            bad_feature_cols.append(col)

    logging.info("Features that don't present in every year: %s", bad_feature_cols)
    df = df.drop(bad_feature_cols, axis=1)

    logging.info("Merging rows by averaging feature values")
    df = df.drop(['year'], axis=1)
    df = df.groupby('propcode').mean().reset_index()

    logging.info(u"Dumping prepared booking-feature matrix: %s", df.shape)
    df.to_csv(args.output_csv, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-p", required=True, dest="property_csv", help=u"Path to a csv file with properties")
    parser.add_argument("-f", required=True, dest="feature_csv", help=u"Path to a csv file with features")
    parser.add_argument('-o', default="property.csv", dest="output_csv",
                        help=u'Path to an output file. Default: property.csv')
    parser.add_argument('-m', default=10, type=int, dest="min_items_per_feature",
                        help=u'Min items per feature. Default: 10')
    parser.add_argument("--log-level", default='INFO', dest="log_level",
                        choices=['DEBUG', 'INFO', 'WARNINGS', 'ERROR'], help=u"Logging level")

    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s', stream=sys.stdout, level=getattr(logging, args.log_level)
    )

    main()
