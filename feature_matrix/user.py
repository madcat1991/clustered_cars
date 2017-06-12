"""
The script prepares user-feature vectors
"""
import argparse
import logging
import sys

import pandas as pd

from feature_matrix.functions import prepare_num_column


def get_bdf():
    bdf = pd.read_csv(args.booking_csv)
    booking_cols = [
        "code", "year", "breakpoint", "propcode",
        "pets", "category", "drivetime", "n_booked_days",
        "avg_spend_per_head"
    ]
    logging.info("Skipped booking columns: %s", set(bdf.columns).difference(booking_cols))
    return bdf[booking_cols]


def get_idf(bdf):
    idf = pd.merge(
        pd.read_csv(args.property_csv),
        bdf[["propcode", "year"]].drop_duplicates(),
        on=["propcode", "year"]
    )
    item_cols = ["propcode", "year", "stars"]
    logging.info("Skipped property columns: %s", set(idf.columns).difference(item_cols))
    idf = idf[item_cols]
    idf.stars = prepare_num_column(idf.stars)
    return idf


def get_udf(bdf):
    udf = pd.read_csv(args.contact_csv)
    user_cols = ["code", "oac_groupdesc"]
    logging.info("Skipped contact columns: %s", set(udf.columns).difference(user_cols))
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
        'broadband',
        'coast 5 miles',
        'complex',
        'countryside views',
        'detached',
        'enclosed garden',
        'enhanced',
        'farm help',
        'fishing - private',
        'golf course nearby - good',
        'high chair',
        'hot tub',
        'indoor pool',
        'jacuzzi',
        'no smoking',
        'on a farm',
        'open fire or woodburner',
        'outdoor heated pool',
        'outdoor unheated pool',
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
        'steam room',
        'tennis court',
        'travel cot',
        'vineyard',
        'wheel chair facilities',
        'dropsided cot',
        'games room',
        'views',
        'audio tour',
        'river or estuary views',
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
    cols_to_binarize = df.columns[df.dtypes == 'object'].drop(["propcode", "code"])
    df = pd.get_dummies(df, columns=cols_to_binarize).fillna(0)

    # dropping columns that don't present in the last year and in less than 50% of the years
    bad_feature_cols = []
    n_years = df.year.unique().size
    max_year = df.year.unique().max()
    for col in df.columns.drop(["code", "propcode", "year"]):
        col_years = df[df[col] > 0].year.unique()
        if max(col_years) != max_year or len(col_years) < n_years * 0.5:
            bad_feature_cols.append(col)
    logging.info("Features that don't present in every year: %s", bad_feature_cols)
    df = df.drop(bad_feature_cols, axis=1)

    df = df.drop(["propcode", "year"], axis=1)
    df["booking_cnt"] = 1  # for future purposes
    df = df.groupby("code").sum().reset_index()
    logging.info("Shape before cleaning: %s", df.shape)

    # dropping columns that can't change anything
    bad_feature_cols = []
    for feature_col in df.columns.drop(["code"]):
        users_per_feature = df.code[df[feature_col] > 0].unique().size
        if users_per_feature < args.min_users_per_feature:
            bad_feature_cols.append(feature_col)

    logging.info("Bad columns: %s", bad_feature_cols)
    df = df.drop(bad_feature_cols, axis=1)

    logging.info(u"Dumping prepared user-feature matrix: %s", df.shape)
    df.to_csv(args.output_csv, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-b", required=True, dest="booking_csv", help=u"Path to a csv file with transformed bookings")
    parser.add_argument("-c", required=True, dest="contact_csv", help=u"Path to a csv file with contacts")
    parser.add_argument("-p", required=True, dest="property_csv", help=u"Path to a csv file with properties")
    parser.add_argument("-f", required=True, dest="feature_csv", help=u"Path to a csv file with features")
    parser.add_argument('-o', default="user.csv", dest="output_csv",
                        help=u'Path to an output file. Default: user.csv')
    parser.add_argument('-m', default=2, type=int, dest="min_users_per_feature",
                        help=u'Min users per feature. Default: 2')
    parser.add_argument("--log-level", default='INFO', dest="log_level",
                        choices=['DEBUG', 'INFO', 'WARNINGS', 'ERROR'], help=u"Logging level")

    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s', stream=sys.stdout, level=getattr(logging, args.log_level)
    )

    main()
