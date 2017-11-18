"""
This script creates descriptions for the property in JSON format
"""

import argparse
import json
import logging

import pandas as pd
import sys


def get_property_full_df():
    logging.info("Preparing the property data frame")
    pdf = pd.read_csv(args.property_csv)
    relevant_years = pdf[["propcode", "year"]].groupby("propcode").year.max().reset_index()
    pdf = pd.merge(relevant_years, pdf, on=["propcode", "year"])

    pfdf = pd.read_csv(args.feature_csv)
    pdf = pd.merge(pdf, pfdf, on=["propcode", "year"], how='left')
    logging.info("The property data frame has been prepared")
    return pdf


address_fields = {
    "region": "region",
    "place": "place",
    "county": "county",
    "postcode": "postcode"
}


space_fields = [
    "bedrooms",
    "sleeps",
    "doubles",
    "twin",
    "bunks",
    "familyrooms",
    "bathrooms",
]

feature_fields = {
    "charges": "charges",
    "heating": "heating",
    "linen": "linen",
    "parking": "parking",
    "stairgate": "stairgate",
    "towels": "towels",
}

possession_field_title = {
    "_19": ("an air conditioner", "air conditioners"),
    "_20": ("a baby sitter", "baby sitters"),
    "barbecue": ("a barbecue", "barbecues"),
    "_24": "big gardens or a farm to wander",
    "_26": "boats or mooring available",
    "broadband": "a broadband internet connection",
    "_30": "a play area for children",
    "_34": "countryside views",
    "_40": ("an enclosed garden", "enclosed gardens"),
    "_42": "equipment for infants",
    "_44": "a private fishing place",
    "_49": "a games room",
    "_58": "a pool",
    "jacuzzi": "a jacuzzi",
    "_66": ("a woodburner", "woodburners"),
    "_67": "a pool",
    "_68": "a pool",
    "piano": "a piano",
    "pool": "a pool",
    "_77": "river or estuary views",
    "sauna": ("a sauna", "saunas"),
    "_82": "sea views",
    "_85": ("a snooker table", "snooker tables"),
    "_88": ("a steam room", "steam rooms"),
    "_90": ("a tennis court", "tennis courts"),
}


true_false_fields = {
    "shortbreakok": "allows short breaks",
    "complex": "a complex",
    "detached": "detached",
    "_52": "good for honeymooners",
    "_64": "non-smoking",
    "_71": "accessible to partially disabled people",
    "pets": "pets-friendly",
    "_100": "accessible to people using wheel chairs"
}


things_nearby_field = {
    "_31": "a coast",
    "_51": "golf courses",
    "_75": "a pub",
    "_76": "a railway",
    "_79": "a sailing",
    "_80": "a sandy beach",
}


def get_address_info(t):
    data = {}
    for field, title in address_fields.items():
        val = getattr(t, field)
        if not pd.isnull(val):
            data[title] = val.strip()
    return data


def get_space_info(t):
    data = {}
    for field in space_fields:
        val = getattr(t, field)
        if val > 0:
            data[field] = int(val)
    return data


def get_possession_info(t):
    data = set()
    for field, title in possession_field_title.items():
        if isinstance(title, tuple):
            singular, plural = title
        else:
             singular = plural = title

        val = getattr(t, field, 0)
        if val == 1:
            data.add(singular)
        elif val > 1:
            data.add(plural)
    return list(data)


def get_nearby_info(t):
    data = set()
    for field, title in things_nearby_field.items():
        if isinstance(title, tuple):
            singular, plural = title
        else:
             singular = plural = title

        val = getattr(t, field, 0)
        if val == 1:
            data.add(singular)
        elif val > 1:
            data.add(plural)
    return list(data)


def get_features_info(t):
    data = {}
    for field, title in feature_fields.items():
        val = getattr(t, field)
        if not pd.isnull(val):
            data[title] = val.strip()
    return data


def get_peculiarities(t):
    data = []
    for field, title in true_false_fields.items():
        if getattr(t, field, 0) > 0:
            data.append(title)
    return data


def collect_property_data(pdf):
    logging.info("Collecting data dicts for properties")
    property_data = {}
    for t in pdf.itertuples():
        propcode = t.propcode
        data = {
            "name": t.pname,
            "stars": t.stars,
            "address": get_address_info(t),
            "space": get_space_info(t),
            "possession": get_possession_info(t),
            "features": get_features_info(t),
            "nearby": get_nearby_info(t),
            "peculiarities": get_peculiarities(t),
        }
        property_data[propcode] = data
    logging.info("Data has been collected")
    return property_data


def main():
    pdf = get_property_full_df()
    property_data = collect_property_data(pdf)

    logging.info("Dumping data to: %s", args.output_path)
    with open(args.output_path, "w") as f:
        json.dump(property_data, f)

    logging.info("Finish")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-p', required=True, dest="property_csv",
                        help='Path to a csv file with properties')
    parser.add_argument('-f', required=True, dest="feature_csv",
                        help='Path to a csv file with property features')
    parser.add_argument('-o', default="property_descrs.json", dest="output_path",
                        help='Path to an output file. Default: property_descrs.json')
    parser.add_argument("--log-level", default='INFO', dest="log_level",
                        choices=['DEBUG', 'INFO', 'WARNINGS', 'ERROR'], help=u"Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s', stream=sys.stdout, level=getattr(logging, args.log_level)
    )

    main()
