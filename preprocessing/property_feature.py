"""
This script cleans and prepares the data set of properties' features for the future usage
"""

import argparse
import logging
import sys

import pandas as pd

from preprocessing.common import raw_data_to_df, check_processed_columns

INTERESTING_COLS = [u'propcode', u'year', u'desc1', u'desc2']

FEATURES_TO_DROP = [
    'arrival time', 'change day',  # no need
    'bovey castle', 'burgh island',  # can be taken from property
    'consider lwl', 'consider sb',  # no need
    'county', 'place', 'region',  # can be taken from property
    'camp bed for 1', 'day bed for 1', 'day bed for 2',
    'offsite entertainment',  # is a combination of others
    'owners website', 'phone',  # no need
    'pets - free of charge', 'pets - unlimited',  # too detailed
    'pvr',
    'suitability',
    'tv'  # is a combination of tele and others
]


YES_NO_COLS = [
    'air conditioning',
    'audio tour',
    'baby sitting',
    'barbecue',
    'barn conversion',
    'big gardens or farm to wander',
    'boats or mooring available',
    'broadband',
    'child restriction',
    'childrens play area',
    'coast 5 miles',
    'complex',
    'countryside views',
    'cycle hire available',
    'detached',
    'enclosed garden',
    'extra infant equipment',
    'farm help',
    'fishing - private',
    'games room',
    'gara rock',
    'golf course nearby - good',
    'good for honeymooners',
    'green cottages',
    'indoor pool',
    'maenporth estate',
    'maid service',
    'no smoking',
    'on a farm',
    'open fire or woodburner',
    'outdoor heated pool',
    'outdoor unheated pool',
    'pamper by the pool',
    'part disabled',
    'pets',
    'piano',
    'pool',
    'pub 1 mile walk',
    'railway 5 miles',
    'river or estuary views',
    'safety deposit box',
    'sailing nearby',
    'sandy beach 1 mile',
    'sea views',
    'shooting',
    'single-storey',
    'snooker table',
    'steam room',
    'tennis court',
    'views',
    'wedding venue',
    'wheel chair facilities',
]

INT_COLS = [
    'blu-ray player',
    'cot',
    'dishwasher',
    'dropsided cot',
    'dvd',
    'freezer',
    'fridge',
    'fridge-freezer',
    'fridge (larder)',
    'high chair',
    'hot tub',
    'jacuzzi',
    'microwave',
    'sauna',
    'sofa bed',
    'tele',
    'travel cot',
    'tumble drier',
    'video',
    'washer-drier',
    'washing machine',
    'zip-linkable'
]

CATEGORICAL_COLS = {
    u"charges", u"enhanced", u"groceries", u"heating",
    u"linen", u"parking", u"towels", u"stairgate", 'vineyard'
}


def process_prop_year_entry(group):
    first = group.iloc[0]
    features = {"propcode": first.propcode, "year": first.year}
    for t in group.itertuples():
        if t.feature in CATEGORICAL_COLS:
            features[t.feature] = u"; ".join(sorted(t.value))
        else:
            features[t.feature] = max(t.value)
    return features


def process_feature_df(df):
    df = df.drop(FEATURES_TO_DROP, axis=1)
    df[YES_NO_COLS] = df[YES_NO_COLS].apply(lambda x: x.str.contains('y', case=False, na=False)).astype(int)
    df[INT_COLS] = df[INT_COLS].apply(lambda x: pd.to_numeric(x, errors='coerce').fillna(0))
    return df


def main():
    df = raw_data_to_df(args.input_csv, delimiter=args.input_csv_delimiter)
    logging.info(u"Data's initial shape: %s", df.shape)

    logging.info(u"Cleaning data")
    df = df[INTERESTING_COLS].rename(columns={u'desc1': u'feature', u'desc2': u'value'})

    # cleaning Year
    df.year = pd.to_numeric(df.year, errors='coerce')
    logging.info(u"Year data has been cleaned")

    # removing NA
    df = df.dropna()
    logging.info(u"Shape after dropping NA: %s", df.shape)

    # lowering
    df.feature = df.feature.apply(lambda x: x.strip().lower())
    df.value = df.value.apply(lambda x: x.strip().lower())

    # checking in advance
    original_features = set(df.feature)
    processed_features = FEATURES_TO_DROP + YES_NO_COLS + INT_COLS + list(CATEGORICAL_COLS)
    check_processed_columns(processed_features, original_features)

    # removing duplicated values of features per a propcode/year pair
    df = df.groupby(['propcode', 'year', 'feature']).value.apply(set).reset_index()

    logging.info(u"Processing data")
    df = pd.DataFrame(
        df.groupby(["propcode", "year"]).apply(process_prop_year_entry).values.tolist()
    )
    df = process_feature_df(df)
    logging.info(u"Shape after cleaning: %s", df.shape)

    logging.info(u"Dumping data to: %s", args.output_csv)
    df.to_csv(args.output_csv, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', required=True, dest="input_csv",
                        help=u"Path to a csv file with properties' features")
    parser.add_argument('--id', default=";", dest="input_csv_delimiter",
                        help=u"The input file's delimiter. Default: ';'")
    parser.add_argument('-o', default="features.csv", dest="output_csv",
                        help=u'Path to an output file. Default: property_feature.csv')
    parser.add_argument("--log-level", default='INFO', dest="log_level",
                        choices=['DEBUG', 'INFO', 'WARNINGS', 'ERROR'], help=u"Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s', stream=sys.stdout, level=getattr(logging, args.log_level)
    )

    main()
