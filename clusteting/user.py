# coding: utf-8

u"""
The script prepares user-feature vectors
"""
import argparse
import logging
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import binarize


RESERVED_COLS = ["code", "booking_cnt"]


def main():
    df = pd.read_csv(args.uf_csv)
    feature_cols = df.columns.drop(RESERVED_COLS)

    logging.info(u"Running through TF-IDF")
    tfidf = TfidfTransformer().fit_transform(df[feature_cols])

    logging.info(u"Clustering via K-Means. Number of clusters: %s", args.n_clusters)
    km = KMeans(n_clusters=args.n_clusters, n_jobs=-1, tol=1e-5).fit(tfidf)

    logging.info(u"Dumping data to: %s", args.output_path)
    with open(args.output_path, "w") as f:
        cnt_per_cluster = pd.Series(km.labels_).value_counts()

        f.write("*** BEGIN INFO ***\n")
        for k, v in cnt_per_cluster.describe().iteritems():
            f.write("%s: %s\n" % (k, v))
        f.write("*** END INFO ***\n")

        for cl_id in xrange(args.n_clusters):
            cluster_features = feature_cols[
                km.cluster_centers_[cl_id].argsort()[-args.n_explainer:]
            ].tolist()
            cluster = df[km.labels_ == cl_id]

            # mean average usage of explanatory features in the cluster
            explanation = cluster[cluster_features].apply(lambda x: x / cluster.booking_cnt).mean()

            f.write("Cluster #%s [%s]\n" % (cl_id, cluster.shape[0]))
            f.write("Explanation:\n")

            for k, v in explanation[explanation > 0.5].iteritems():
                f.write("-> %s: %.3f\n" % (k, v))

            f.write("Users:\n")
            f.write("%s\n" % ", ".join(cluster.code.tolist()))
            f.write("---\n")
    logging.info(u"Finish")


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    # parser.add_argument("-u", required=True, dest="uf_csv", help=u"Path to a user-feature csv file")
    # parser.add_argument("-n", default=500, dest="n_cluster",
    #                     help=u"Number of clusters to produce. Default: 500")
    # parser.add_argument("-e", default=6, dest="n_explainer",
    #                     help=u"Number of explanatory features per cluster. Default: 6")
    # parser.add_argument('-o', default="user.txt", dest="output_path",
    #                     help=u'Path to an output file. Default: user_clusters.txt')
    # parser.add_argument("--log-level", default='INFO', dest="log_level",
    #                     choices=['DEBUG', 'INFO', 'WARNINGS', 'ERROR'], help=u"Logging level")
    #
    # args = parser.parse_args()

    from collections import namedtuple

    args = namedtuple(
        "args",
        ["uf_csv", "n_clusters", "n_explainer", "output_path"]
    )

    args.uf_csv = '/Users/user/PyProjects/clustered_cars/data/featured/user.csv'
    args.n_clusters = 500
    args.n_explainer = 10
    args.output_path = '/Users/user/PyProjects/clustered_cars/data/clustered/user.txt'
    args.log_level = 'INFO'

    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s', stream=sys.stdout, level=getattr(logging, args.log_level)
    )

    main()
