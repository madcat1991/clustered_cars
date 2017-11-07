import logging

import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix
from sklearn.preprocessing import binarize, normalize

logger = logging.getLogger(__name__)


class ClusterRecommender(object):
    def __init__(self, ug_bg_recs_m, user_dp, booking_dp, item_dp):
        self.ug_bg_recs_m = ug_bg_recs_m
        self.booking_dp =booking_dp
        self.user_dp = user_dp
        self.item_dp = item_dp

    def get_recs(self, ug_id, iid_recs, top_clusters=None, min_iid_per_bg=None):
        bg_recs_row = self.ug_bg_recs_m[ug_id]
        bg_mask = binarize(
            self.item_dp.get_iid_per_bg_row(binarize(iid_recs), min_iid_per_bg)
        )
        bg_recs_row = bg_recs_row.multiply(bg_mask)

        if top_clusters is not None:
            arg_ids = np.argsort(bg_recs_row.data)[-top_clusters:]
            rows, cols = bg_recs_row.nonzero()
            bg_recs_row = csr_matrix(
                (bg_recs_row.data[arg_ids], (rows[arg_ids], cols[arg_ids])),
                shape=bg_recs_row.shape
            )
        return bg_recs_row

    @staticmethod
    def load(config, user_dp, booking_dp, item_dp):
        recs_m_path = config['UG_BG_RECS_MATRIX_PATH']

        ug_bg_recs_m = mmread(recs_m_path).tocsr()
        logger.info("Cluster recs matrix has been load")

        return ClusterRecommender(ug_bg_recs_m, user_dp, booking_dp, item_dp)


class PopItemRecommender(object):
    def __init__(self, booking_dp, item_dp):
        self.booking_dp = booking_dp
        self.item_dp = item_dp

    def get_recs(self, uid, top_items=None):
        active_iids = self.item_dp.get_active_iids()
        booked_iids = self.booking_dp.get_iids_for_uid(uid)
        if booked_iids:
            active_iids = active_iids.difference(booked_iids)

        obs_per_iid = self.booking_dp.get_obs_per_iid()
        obs_per_iid = obs_per_iid[obs_per_iid.index.isin(active_iids)]
        unobserved_iids = list(active_iids.difference(obs_per_iid.index))

        # iids available for recs
        iids = np.r_[obs_per_iid.index, unobserved_iids]
        # scores of iids
        scores = np.r_[obs_per_iid.values, [1e-6] * len(unobserved_iids)]  # constant to guarantee nnz

        arg_ids = np.argsort(scores)[-top_items:][::-1] if top_items else np.argsort(scores)[::-1]
        recs = self.item_dp.get_score_per_iid_row(iids[arg_ids], scores[arg_ids])
        return recs

    @staticmethod
    def load(booking_dp, item_dp):
        return PopItemRecommender(booking_dp, item_dp)


class CBItemRecommender(object):
    def __init__(self, booking_dp, item_dp):
        self.booking_dp = booking_dp
        self.item_dp = item_dp

    def get_recs(self, uid, top_items=None):
        active_iids = self.item_dp.get_active_iids()
        booked_iids = self.booking_dp.get_iids_for_uid(uid)
        if booked_iids:
            active_iids = active_iids.difference(booked_iids)

        active_iids = np.array(active_iids)
        uf_m = normalize(self.booking_dp.get_uids_feature_matrix([uid]))
        if_m = normalize(self.booking_dp.get_iids_feature_matrix(active_iids))
        scores = uf_m.dot(if_m.T).todense().A1

        arg_ids = np.argsort(scores)[-top_items:][::-1] if top_items else np.argsort(scores)[::-1]
        recs = self.item_dp.get_score_per_iid_row(active_iids[arg_ids], scores[arg_ids])
        return recs

    @staticmethod
    def load(booking_dp, item_dp):
        return CBItemRecommender(booking_dp, item_dp)
