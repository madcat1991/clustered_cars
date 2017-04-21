import argparse
import logging
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation


RESERVED_COLS = ["bookcode", "propcode"]


def add_topic_features_to_tree(tree, topic_features):
    topic_features = sorted(topic_features)
    head, *tail = topic_features
    subtree = tree.setdefault(head, {})
    if tail:
        add_topic_features_to_tree(subtree, tail)


def get_clusters_features_from_tree(tree, prefix=None):
    result = []
    for ftr, subtree in tree.items():
        new_prefix = prefix + [ftr] if prefix else [ftr]
        if subtree:
            result += get_clusters_features_from_tree(subtree, new_prefix)
        else:
            result.append(new_prefix)
    return result


def get_cluster_features(lda, bdf, feature_columns, min_features_per_topic, min_items_per_topic):
    logging.info("Identifying clusters")
    features_tree = {}
    for i, topic in enumerate(lda.components_):
        if i % 100 == 0:
            logging.info("%s topics have been processed", i)

        topic_df = None
        topic_features = []
        border_value = np.percentile(topic, 95)
        for col_id in np.argsort(topic)[::-1]:
            if topic[col_id] < border_value:
                break

            ftr = feature_columns[col_id]
            topic_df = bdf[bdf[ftr] > 0] if topic_df is None else topic_df[topic_df[ftr] > 0]

            if topic_df.propcode.unique().size >= min_items_per_topic:
                topic_features.append(ftr)
            else:
                break

        if len(topic_features) >= min_features_per_topic:
            add_topic_features_to_tree(features_tree, topic_features)

    cluster_features = get_clusters_features_from_tree(features_tree)
    logging.info("Number of cluster: %s", len(cluster_features))
    return cluster_features


def get_clusters(bdf, cluster_features, min_ftr_p=0.7):
    logging.info("Collecting clusters")
    clusters = {}
    for cl_id, features in enumerate(cluster_features):
        mask = np.all([bdf[ftr] == 1 for ftr in features], axis=0)
        cdf = bdf[mask]

        booking_ids = cdf.bookcode.unique().tolist()
        item_ids = cdf.propcode.unique().tolist()

        p_per_ftr = cdf.drop(RESERVED_COLS, axis=1).mean()
        desc_ftr = {ftr: score for ftr, score in p_per_ftr[p_per_ftr >= min_ftr_p].iteritems()}

        clusters[cl_id] = {
            "bookings": booking_ids,
            "items": item_ids,
            "features": desc_ftr
        }
    logging.info("Clusters have been collected")
    return clusters


def get_statistics(bdf, clusters):
    logging.info("Calculating statistics")
    available_bookings = set(bdf.bookcode)
    available_items = set(bdf.propcode)

    covered_bookings = Counter()
    covered_items = Counter()

    cluster_length_bookings = []
    cluster_length_items = []

    for cl_id, cl_info in clusters.items():
        covered_bookings.update(cl_info["bookings"])
        covered_items.update(cl_info["items"])

        cluster_length_bookings.append(len(cl_info["bookings"]))
        cluster_length_items.append(len(cl_info["items"]))

    booking_coverage = \
        len(available_bookings.intersection(covered_bookings)) / len(available_bookings)
    item_coverage = \
        len(available_items.intersection(covered_items)) / len(available_items)

    stats = {
        "booking_coverage": booking_coverage,
        "item_coverage": item_coverage,
        "booking_intersection_hist": np.histogram(list(covered_bookings.values())),
        "item_intersection_hist": np.histogram(list(covered_items.values()))
    }
    for k, v in stats.items():
        print(k, v)

    print("Bookings")
    print(pd.Series(cluster_length_bookings).describe())

    print("Items")
    print(pd.Series(cluster_length_items).describe())
    return stats


def main():
    logging.info("Reading the csv file with bookings")
    bdf = pd.read_csv(args.bf_csv)
    logging.info("Bookings dataset shape: %s", bdf.shape)
    feature_part = bdf.drop(RESERVED_COLS, axis=1)

    # logging.info("Grouping data by propcode")
    # gdf = bdf.groupby("propcode").mean()
    # logging.info("Resulting shape: %s", gdf.shape)

    logging.info("Learning LDA model")
    lda = LatentDirichletAllocation(
        n_topics=args.n_topics, learning_method='batch', verbose=1
    ).fit(feature_part)

    cluster_features = get_cluster_features(
        lda,
        bdf,
        feature_part.columns,
        args.min_features_per_topic,
        args.min_items_per_topic
    )
    clusters = get_clusters(bdf, cluster_features)
    stats = get_statistics(bdf, clusters)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-b", dest="bf_csv", #required=True,
                        default="/Users/tural/PyProjects/clustered_cars/data/featured/bookings.csv",
                        help=u"Path to a booking-feature csv file")
    parser.add_argument("-t", default=1000, dest="n_topics", type=int,
                        help=u"Number of topics to produce. "
                        u"This value is not equal to the number of resulting clusters. Default: 1000")
    parser.add_argument("-f", default=5, dest="min_features_per_topic", type=int,
                        help=u"Minimum number of features per a topics. Default: 5")
    parser.add_argument("-i", default=10, dest="min_items_per_topic", type=int,
                        help=u"Minimum number of items per a topics. Default: 10")
    parser.add_argument('-o', default="bookings.txt", dest="output_path",
                        help=u'Path to an output file. Default: bookings.txt')
    parser.add_argument("--log-level", default='INFO', dest="log_level",
                        choices=['DEBUG', 'INFO', 'WARNINGS', 'ERROR'], help=u"Logging level")

    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s', stream=sys.stdout, level=getattr(logging, args.log_level)
    )

    main()