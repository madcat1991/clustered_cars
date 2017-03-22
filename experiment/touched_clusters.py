# coding: utf-8
import logging
import sys

import numpy as np
import pandas as pd
import pickle

from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import normalize


def get_bg_data():
    bid_to_bg = {}
    bg_iids = {}
    with open(args.booking_cluster) as f:
        # skipping
        while not f.next().startswith("Cluster"):
            pass

        cl_id = 0
        for line in f:
            if line.startswith("Bookings:"):
                for bid in line.lstrip("Bookings:").split(","):
                    bid_to_bg[bid.strip()] = cl_id
            elif line.startswith("Items:"):
                bg_iids[cl_id] = {iid.strip() for iid in line.lstrip("Items:").split(",")}
            elif line.startswith("Cluster"):
                cl_id += 1
    return bid_to_bg, bg_iids


def process(bdf, bgdf, bidf, ui_iid_recs):
    res = {}

    for i, (key, iid_recs) in enumerate(ui_iid_recs.iteritems()):
        if i % 100 == 0:
            logging.info(u"Processed %s", i)

        iids_and_bgs = bdf[["propcode", "cl_id"]][bdf.propcode.isin(iid_recs)].drop_duplicates()
        iids_and_bgs = iids_and_bgs.groupby("propcode").apply(lambda x: x.cl_id.tolist())

        touched = []
        for iid in iid_recs:
            candidate_bgs = iids_and_bgs.loc[iid]
            if len(candidate_bgs) > 1:
                bg_m = bgdf.loc[candidate_bgs]
                i_v = bidf.loc[iid].values.reshape(1, -1)
                dist = euclidean_distances(normalize(i_v), normalize(bg_m))[0]
                bg = candidate_bgs[np.argmin(dist)]
            else:
                bg = candidate_bgs[0]

            touched.append(bg)
        res[key] = touched
    return res


def main():
    bdf = pd.read_csv(args.booking_features)
    bid_to_bg, bg_iids = get_bg_data()

    with open(args.ui_iid_recs) as f:
        ui_iid_recs = pickle.load(f)

    bdf["cl_id"] = bdf.apply(lambda x: bid_to_bg[x.bookcode], axis=1)
    bgdf = bdf.drop(["propcode", "bookcode"], axis=1).groupby("cl_id").sum()
    bidf = bdf.drop(["bookcode", "cl_id"], axis=1).groupby("propcode").sum()

    res = process(bdf, bgdf, bidf, ui_iid_recs)
    with open(args.output, "w") as f:
        pickle.dump(res, f)


if __name__ == '__main__':
    from collections import namedtuple

    args = namedtuple(
        "args",
        [
            "booking_features", "booking_cluster", "log_level",
            "ui_iid_recs", "output"
        ]
    )

    args.ui_iid_recs = "/Users/user/PyProjects/clustered_cars/experiment/ui_iid_recs.pkl"
    args.output = "/Users/user/PyProjects/clustered_cars/experiment/ui_iid_based_bg_recs.pkl"
    args.booking_features = "/Users/user/PyProjects/clustered_cars/data/featured/booking.csv"
    args.booking_cluster = '/Users/user/PyProjects/clustered_cars/data/clustered/bookings.txt'
    args.log_level = 'INFO'

    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s', stream=sys.stdout, level=getattr(logging, args.log_level)
    )

    main()
