# coding: utf-8

u"""
The script clusters bookings
"""
import argparse
import logging
import sys

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

RESERVED_COLS = ["bookcode", "propcode"]


def main():
    df = pd.read_csv(args.bf_csv)
    feature_cols = df.columns.drop(RESERVED_COLS)

    logging.info(u"Running PCA")
    feature_part = PCA(n_components=args.n_components).fit_transform(df[feature_cols])

    logging.info(u"Clustering via K-Means. Number of clusters: %s", args.n_clusters)
    km = KMeans(n_clusters=args.n_clusters, tol=1e-5).fit(feature_part)

    logging.info(u"Dumping data to: %s", args.output_path)
    with open(args.output_path, "w") as f:
        bookings_per_cluster = pd.Series(km.labels_).value_counts()
        f.write("*** BOOKINGS INFO ***\n")
        for k, v in bookings_per_cluster.describe().iteritems():
            f.write("%s: %s\n" % (k, v))
        f.write("***\n")

        items_per_cluster = pd.Series({
            cl_id: df.propcode[km.labels_ == cl_id].unique().size
            for cl_id in xrange(args.n_clusters)
        })
        f.write("*** ITEMS INFO ***\n")
        for k, v in items_per_cluster.describe().iteritems():
            f.write("%s: %s\n" % (k, v))
        f.write("***\n")

        for cl_id in xrange(args.n_clusters):
            cluster = df[km.labels_ == cl_id]

            # average usage of explanatory features in the cluster
            explanation = cluster[feature_cols].mean()

            f.write(
                "Cluster #%s [%s | %s]\n" %
                (cl_id, bookings_per_cluster[cl_id], items_per_cluster[cl_id])
            )
            f.write("Explanation:\n")

            for k, v in explanation[explanation > 0.8].iteritems():
                f.write("-> %s: %.3f\n" % (k, v))

            f.write("Bookings:\n")
            f.write("%s\n" % ", ".join(cluster.bookcode.tolist()))
            f.write("Items:\n")
            f.write("%s\n" % ", ".join(cluster.propcode.unique().tolist()))
            f.write("---\n")
    logging.info(u"Finish")


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    # parser.add_argument("-b", required=True, dest="bf_csv", help=u"Path to a booking-feature csv file")
    # parser.add_argument("-n", default=200, dest="n_cluster",
    #                     help=u"Number of clusters to produce. Default: 200")
    # parser.add_argument("-p", default=25, dest="n_components",
    #                     help=u"Number of PCA components. Default: 25")
    # parser.add_argument('-o', default="user.txt", dest="output_path",
    #                     help=u'Path to an output file. Default: booking.txt')
    # parser.add_argument("--log-level", default='INFO', dest="log_level",
    #                     choices=['DEBUG', 'INFO', 'WARNINGS', 'ERROR'], help=u"Logging level")
    #
    # args = parser.parse_args()

    from collections import namedtuple

    args = namedtuple(
        "args",
        ["bf_csv", "n_clusters", "n_components", "output_path", "log_level"]
    )

    args.bf_csv = '/Users/user/PyProjects/clustered_cars/data/featured/booking.csv'
    args.n_clusters = 450
    args.n_components = 20
    args.output_path = '/Users/user/PyProjects/clustered_cars/data/clustered/booking.txt'
    args.log_level = 'INFO'

    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s', stream=sys.stdout, level=getattr(logging, args.log_level)
    )

    main()
