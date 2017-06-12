"""
The script clusters bookings using faiss package
"""

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from clusteting.method import smart_kmeans_clustering

RESERVED_COLS = ["bookcode", "propcode", "year"]


def main():
    df = pd.read_csv(args.bf_csv)
    feature_cols = df.columns.drop(RESERVED_COLS)
    m = np.ascontiguousarray(df[feature_cols].values).astype('float32')

    logging.info(u"Clustering via K-Means")
    _, cluster_labels = smart_kmeans_clustering(
        m, df.propcode, args.n_clusters, args.min_props_per_cluster
    )
    df["cl_id"] = cluster_labels

    logging.info(u"Dumping data to: %s", args.output_path)
    with open(args.output_path, "w") as f:

        f.write("*** BOOKINGS INFO ***\n")
        booking_per_cluster = df.groupby("cl_id").bookcode.nunique()
        for k, v in booking_per_cluster.describe().iteritems():
            f.write("%s: %s\n" % (k, v))
        f.write("***\n")

        f.write("*** ITEMS INFO ***\n")
        items_per_cluster = df.groupby("cl_id").propcode.nunique()
        for k, v in items_per_cluster.describe().iteritems():
            f.write("%s: %s\n" % (k, v))
        f.write("***\n")

        for cl_id in sorted(df.cl_id.unique()):
            cluster = df[df.cl_id == cl_id]

            # average usage of explanatory features in the cluster
            explanation = cluster[feature_cols].mean()

            f.write(
                "Cluster #%s [%s | %s]\n" %
                (cl_id, booking_per_cluster[cl_id], items_per_cluster[cl_id])
            )
            f.write("Explanation:\n")

            for k, v in explanation[explanation > 0.7].iteritems():
                f.write("-> %s: %.3f\n" % (k, v))

            f.write("Bookings: %s\n" % ", ".join(cluster.bookcode.tolist()))
            f.write("Items: %s\n" % ", ".join(cluster.propcode.unique().tolist()))
            f.write("---\n")
    logging.info(u"Finish")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-b", required=True, dest="bf_csv", help=u"Path to a booking-feature csv file")
    parser.add_argument("-n", default=1000, dest="n_clusters", type=int,
                        help=u"Initial number of clusters for KMeans. Default: 1000")
    parser.add_argument("-m", default=10, dest="min_props_per_cluster", type=int,
                        help=u"Min number of properties per cluster. Default: 10")
    parser.add_argument('-o', default="bookings.txt", dest="output_path",
                        help=u'Path to an output file. Default: bookings.txt')
    parser.add_argument("--log-level", default='INFO', dest="log_level",
                        choices=['DEBUG', 'INFO', 'WARNINGS', 'ERROR'], help=u"Logging level")

    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s', stream=sys.stdout, level=getattr(logging, args.log_level)
    )

    main()
