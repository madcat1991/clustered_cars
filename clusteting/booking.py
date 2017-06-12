"""
The script clusters bookings using faiss package
"""

import argparse
import logging
import sys

import numpy as np
import pandas as pd

import faiss

RESERVED_COLS = ["bookcode", "propcode"]


def main():
    df = pd.read_csv(args.bf_csv)
    feature_cols = df.columns.drop(RESERVED_COLS)
    m = np.ascontiguousarray(df[feature_cols].values).astype('float32')

    logging.info(u"Clustering via K-Means. Number of clusters: %s", args.n_clusters)
    kmeans = faiss.Kmeans(m.shape[1], args.n_clusters, verbose=True)
    kmeans.train(m)

    logging.info("Assigning bookings to clusters")
    D, I = kmeans.index.search(m, 1)
    labels = I.reshape(-1)

    logging.info(u"Dumping data to: %s", args.output_path)
    with open(args.output_path, "w") as f:
        bookings_per_cluster = pd.Series(labels).value_counts()
        f.write("*** BOOKINGS INFO ***\n")
        for k, v in bookings_per_cluster.describe().iteritems():
            f.write("%s: %s\n" % (k, v))
        f.write("***\n")

        items_per_cluster = pd.Series({
            cl_id: df.propcode[labels == cl_id].unique().size
            for cl_id in range(args.n_clusters)
        })
        f.write("*** ITEMS INFO ***\n")
        for k, v in items_per_cluster.describe().iteritems():
            f.write("%s: %s\n" % (k, v))
        f.write("***\n")

        for cl_id in range(args.n_clusters):
            cluster = df[labels == cl_id]

            # average usage of explanatory features in the cluster
            explanation = cluster[feature_cols].mean()

            f.write(
                "Cluster #%s [%s | %s]\n" %
                (cl_id, bookings_per_cluster[cl_id], items_per_cluster[cl_id])
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
    parser.add_argument("-n", default=400, dest="n_clusters", type=int,
                        help=u"Number of clusters to produce. Default: 400")
    parser.add_argument('-o', default="bookings.txt", dest="output_path",
                        help=u'Path to an output file. Default: bookings.txt')
    parser.add_argument("--log-level", default='INFO', dest="log_level",
                        choices=['DEBUG', 'INFO', 'WARNINGS', 'ERROR'], help=u"Logging level")

    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s', stream=sys.stdout, level=getattr(logging, args.log_level)
    )

    main()
