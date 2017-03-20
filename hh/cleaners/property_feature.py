# coding: utf-8

u"""
This script cleans and prepares the data set of properties' features for the future usage
"""

import argparse
import logging
import sys

import pandas as pd

INTERESTING_COLS = [u'PropCode', u'Year', u'Desc1', u'Desc2']

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
    'detached',
    'enclosed garden',
    'farm help',
    'fishing - private',
    'games room',
    'gara rock',
    'golf course nearby - good',
    'good for honeymooners',
    'green cottages',
    'indoor pool',
    'maenporth estate',
    'no smoking',
    'on a farm',
    'open fire or woodburner',
    'outdoor heated pool',
    'outdoor unheated pool',
    'part disabled',
    'pets',
    'piano',
    'pool',
    'pub 1 mile walk',
    'railway 5 miles',
    'river or estuary views',
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
    u"linen", u"parking", u"towels", u"stairgate"
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
    df[YES_NO_COLS] = df[YES_NO_COLS].apply(lambda x: x.str.contains('y')).fillna(False).astype(int)
    df[INT_COLS] = df[INT_COLS].apply(lambda x: pd.to_numeric(x, errors='coerce').fillna(0))
    return df


def main():
    df = pd.read_csv(args.input_csv, delimiter=args.input_csv_delimiter)
    logging.info(u"Data's initial shape: %s", df.shape)

    logging.info(u"Cleaning data")
    df = df[INTERESTING_COLS].rename(columns={
        u'PropCode': u"propcode", u'Year': u"year",
        u'Desc1': u'feature', u'Desc2': u'value'
    })

    # cleaning Year
    df.year = pd.to_numeric(df.year, errors='coerce')
    logging.info(u"Year data has been cleaned")

    # removing NA
    df = df.dropna()

    # lowering
    df.feature = df.feature.apply(lambda x: x.strip().lower())
    df.value = df.value.apply(lambda x: x.strip().lower())

    # removing duplicated values of features per a propcode/year pair
    df = df.groupby(['propcode', 'year', 'feature']).value.apply(set).reset_index()
    logging.info(u"Shape after cleaning: %s", df.shape)

    logging.info(u"Processing data")
    df = pd.DataFrame(
        df.groupby(["propcode", "year"]).apply(process_prop_year_entry).values.tolist()
    )
    df = process_feature_df(df)

    logging.info(u"Dumping data to: %s", args.output_csv)
    df.to_csv(args.output_csv, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', required=True, dest="input_csv",
                        help=u"Path to a csv file with properties' features")
    parser.add_argument('--id', default=";", dest="input_csv_delimiter",
                        help=u"The input file's delimiter. Default: ';'")
    parser.add_argument('-o', default="features.csv", dest="output_csv",
                        help=u'Path to an output file. Default: HH_Cleaned_Features.csv')
    parser.add_argument("--log-level", default='INFO', dest="log_level",
                        choices=['DEBUG', 'INFO', 'WARNINGS', 'ERROR'], help=u"Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s', stream=sys.stdout, level=getattr(logging, args.log_level)
    )

    main()
