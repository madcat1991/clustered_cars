"""
The script clusters users
"""

import argparse
import logging
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer

import faiss

RESERVED_COLS = ["code", "booking_cnt"]


def main():
    df = pd.read_csv(args.uf_csv)
    feature_cols = df.columns.drop(RESERVED_COLS)

    logging.info(u"Running TF-IDF")
    tfidf = TfidfTransformer().fit_transform(df[feature_cols])
    m = np.ascontiguousarray(tfidf.todense()).astype('float32')

    logging.info(u"Clustering via K-Means. Number of clusters: %s", args.n_clusters)
    kmeans = faiss.Kmeans(m.shape[1], args.n_clusters, verbose=True)
    kmeans.train(m)

    logging.info("Assigning users to clusters")
    D, I = kmeans.index.search(m, 1)
    labels = I.reshape(-1)

    logging.info(u"Dumping data to: %s", args.output_path)
    with open(args.output_path, "w") as f:
        cnt_per_cluster = pd.Series(labels).value_counts()

        f.write("*** BEGIN INFO ***\n")
        for k, v in cnt_per_cluster.describe().iteritems():
            f.write("%s: %s\n" % (k, v))
        f.write("*** END INFO ***\n")

        for cl_id in range(args.n_clusters):
            cluster = df[labels == cl_id]

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
    parser.add_argument('-o', default="users.txt", dest="output_path",
                        help=u'Path to an output file. Default: users.txt')
    parser.add_argument("--log-level", default='INFO', dest="log_level",
                        choices=['DEBUG', 'INFO', 'WARNINGS', 'ERROR'], help=u"Logging level")

    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s', stream=sys.stdout, level=getattr(logging, args.log_level)
    )

    main()
