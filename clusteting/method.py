import logging
import numpy as np
import pandas as pd
import faiss


def smart_kmeans_clustering(X, obj_Y, n_clusters, min_obj_per_cluster=5, search_in=20):
    logging.info(
        u"Params: initial number of clusters: %s, min objects per cluster: %s",
        n_clusters, min_obj_per_cluster
    )

    kmeans = faiss.Kmeans(X.shape[1], n_clusters, verbose=True)
    kmeans.train(X)
    D, I = kmeans.index.search(X, 1)

    cluster_labels = I.reshape(-1)
    dists = D[:, 0].reshape(-1)

    cum_bad_cluster_ids = np.array([])
    for n_obj_per_cl in range(1, min_obj_per_cluster):
        df = pd.DataFrame({"obj_id": obj_Y, "cl_id": cluster_labels})
        obj_per_cluster = df.groupby("cl_id").obj_id.nunique()
        bad_cluster_ids = obj_per_cluster[obj_per_cluster == n_obj_per_cl].index

        if len(bad_cluster_ids) > 0:
            logging.info("Number of cluster with %s elements = %s", n_obj_per_cl, len(bad_cluster_ids))

            cum_bad_cluster_ids = np.r_[cum_bad_cluster_ids, bad_cluster_ids]

            # we're changing both cluster_labels and dists, it's easier to work with row_ids
            bad_row_ids = np.where(np.in1d(cluster_labels, cum_bad_cluster_ids))[0]
            bad_X = X[bad_row_ids, :]
            D, I = kmeans.index.search(bad_X, search_in)
            mask_m = ~np.in1d(I, cum_bad_cluster_ids).reshape(bad_X.shape[0], -1)

            new_labels = []
            new_dists = []
            for row_id in range(bad_X.shape[0]):
                row_mask = mask_m[row_id, :]
                new_labels.append(I[row_id, :][row_mask][0])
                new_dists.append(D[row_id, :][row_mask][0])

            cluster_labels[bad_row_ids] = new_labels
            dists[bad_row_ids] = new_dists

    final_n_clusters = np.unique(cluster_labels).size
    logging.info("Final number of clusters %s", final_n_clusters)

    # reindexing
    ix = {}
    cluster_labels = np.array([ix.setdefault(el, len(ix)) for el in cluster_labels])
    return dists, cluster_labels
