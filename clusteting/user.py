# coding: utf-8

u"""
The script clusters users
"""
import argparse
import logging
import sys

import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfTransformer


RESERVED_COLS = ["code", "booking_cnt"]


def main():
    df = pd.read_csv(args.uf_csv)
    feature_cols = df.columns.drop(RESERVED_COLS)

    logging.info(u"Running TF-IDF")
    tfidf = TfidfTransformer().fit_transform(df[feature_cols])

    logging.info(u"Running PCA")
    feature_part = PCA(n_components=args.n_components).fit_transform(tfidf.todense())

    logging.info(u"Clustering via K-Means. Number of clusters: %s", args.n_clusters)
    km = MiniBatchKMeans(
        n_init=5,
        batch_size=int(feature_part.shape[0] * 0.4),
        n_clusters=args.n_clusters,
        verbose=1
    ).fit(feature_part)
    # logging.info(u"Clustering via Birch. Number of clusters: %s", args.n_clusters)
    # km = Birch(n_clusters=args.n_clusters, threshold=0.1, branching_factor=100).fit(feature_part)

    logging.info(u"Dumping data to: %s", args.output_path)
    with open(args.output_path, "w") as f:
        cnt_per_cluster = pd.Series(km.labels_).value_counts()

        f.write("*** BEGIN INFO ***\n")
        for k, v in cnt_per_cluster.describe().iteritems():
            f.write("%s: %s\n" % (k, v))
        f.write("*** END INFO ***\n")

        for cl_id in xrange(args.n_clusters):
            cluster = df[km.labels_ == cl_id]

            # mean average usage of explanatory features in the cluster
            explanation = cluster[feature_cols].apply(lambda x: x / cluster.booking_cnt).mean()

            f.write("Cluster #%s [%s]\n" % (cl_id, cluster.shape[0]))
            f.write("Explanation:\n")

            for k, v in explanation[explanation > 0.7].iteritems():
                f.write("-> %s: %.3f\n" % (k, v))

            f.write("Users: %s\n" % ", ".join(cluster.code.tolist()))
            f.write("---\n")
    logging.info(u"Finish")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-u", required=True, dest="uf_csv", help=u"Path to a user-feature csv file")
    parser.add_argument("-n", default=900, dest="n_clusters", type=int,
                        help=u"Number of clusters to produce. Default: 900")
    parser.add_argument("-c", default=50, dest="n_components", type=int,
                        help=u"Number of PCA components to consider. Default: 50")
    parser.add_argument('-o', default="user.txt", dest="output_path",
                        help=u'Path to an output file. Default: user.txt')
    parser.add_argument("--log-level", default='INFO', dest="log_level",
                        choices=['DEBUG', 'INFO', 'WARNINGS', 'ERROR'], help=u"Logging level")

    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s', stream=sys.stdout, level=getattr(logging, args.log_level)
    )

    main()
