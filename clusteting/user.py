"""
The script clusters users
"""

import argparse
import logging
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer

from clusteting.method import smart_kmeans_clustering

RESERVED_COLS = ["code", "booking_cnt"]
FEATURE_THRESHOLD = 0.6


def main():
    df = pd.read_csv(args.uf_csv)
    feature_cols = df.columns.drop(RESERVED_COLS)

    logging.info(u"Running TF-IDF")
    tfidf = TfidfTransformer().fit_transform(df[feature_cols])
    m = np.ascontiguousarray(tfidf.todense()).astype('float32')

    logging.info(u"Clustering via K-Means")
    _, cluster_labels = smart_kmeans_clustering(
        m, df.code, args.n_clusters, args.min_props_per_cluster
    )

    logging.info(u"Dumping data to: %s", args.output_path)
    with open(args.output_path, "w") as f:
        cnt_per_cluster = pd.Series(cluster_labels).value_counts()

        f.write("*** BEGIN INFO ***\n")
        for k, v in cnt_per_cluster.describe().iteritems():
            f.write("%s: %s\n" % (k, v))
        f.write("*** END INFO ***\n")

        for cl_id in range(args.n_clusters):
            cluster = df[cluster_labels == cl_id]

            # mean average usage of explanatory features in the cluster
            explanation = cluster[feature_cols].apply(lambda x: x / cluster.booking_cnt).mean()

            f.write("Cluster #%s [%s]\n" % (cl_id, cluster.shape[0]))
            f.write("Explanation:\n")

            for k, v in explanation[explanation > FEATURE_THRESHOLD].iteritems():
                f.write("-> %s: %.3f\n" % (k, v))

            f.write("Users: %s\n" % ", ".join(cluster.code.tolist()))
            f.write("---\n")
    logging.info(u"Finish")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-u", required=True, dest="uf_csv", help=u"Path to a user-feature csv file")
    parser.add_argument("-n", default=1200, dest="n_clusters", type=int,
                        help=u"Initial number of clusters for KMeans. Default: 1200")
    parser.add_argument("-m", default=5, dest="min_props_per_cluster", type=int,
                        help=u"Min number of users per cluster. Default: 5")
    parser.add_argument('-o', default="user.txt", dest="output_path",
                        help=u'Path to an output file. Default: user.txt')
    parser.add_argument("--log-level", default='INFO', dest="log_level",
                        choices=['DEBUG', 'INFO', 'WARNINGS', 'ERROR'], help=u"Logging level")

    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s', stream=sys.stdout, level=getattr(logging, args.log_level)
    )

    main()
