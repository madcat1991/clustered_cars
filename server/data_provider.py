import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from misc.common import get_ug_data, get_bg_data, get_group_features


FEATURE_THRESHOLD = 0.5


class UserDataProvider(object):
    def __init__(self, udf, uid_to_ug, ug_features):
        # TODO probably later *df-s should be removed from constructors
        self._uid_to_ug = uid_to_ug
        self._ug_features = ug_features
        self._prepare_uid_features(udf)

    def _prepare_uid_features(self, udf):
        booking_cnt = udf.booking_cnt
        udf = udf.drop(["booking_cnt"], axis=1).set_index("code")
        udf = udf.apply(lambda x: x / booking_cnt)
        udf[udf < FEATURE_THRESHOLD] = 0
        self._uid_features = udf

    def get_cluster_id(self, uid):
        return self._uid_to_ug.get(uid)

    def get_uid_features(self, uid):
        return self._uid_features.loc[uid] if uid in self._uid_features else {}

    def get_cluster_features(self, cluster_id):
        return self._ug_features.get(cluster_id, {})

    @staticmethod
    def load(config):
        uid_to_ug = get_ug_data(config['UG_FILE_PATH'])
        ug_features = get_group_features(config['UG_FILE_PATH'])
        udf = pd.read_csv(config['USER_FEATURE_FILE_PATH'])
        return UserDataProvider(udf, uid_to_ug, ug_features)


class BookingDataProvider(object):
    def __init__(self, bdf, bg_features):
        self._bg_features = bg_features
        self._prepare_booking_iid_uid_part(bdf)
        self._prepare_obs_per_iid(bdf)
        self._prepare_uid_booking_summaries(bdf)

    def _prepare_obs_per_iid(self, bdf):
        self._obs_per_iid = bdf.groupby('propcode').bookcode.nunique()

    def _prepare_booking_iid_uid_part(self, bdf):
        cols = ["bookcode", "propcode", "code"]
        self._b_iid_uid = bdf[cols]

    def _prepare_uid_booking_summaries(self, bdf):
        cols = ["bookcode", "propcode", "year"]
        data = bdf.drop(cols, axis=1)
        data = data.groupby("code").mean()
        data[data < FEATURE_THRESHOLD] = 0
        self._uid_booking_summaries = data

    def get_iids_for_uid(self, uid):
        return set(self._b_iid_uid[self._b_iid_uid.code == uid].propcode)

    def get_obs_per_iid(self):
        return self._obs_per_iid

    def get_cluster_features(self, cluster_id):
        return self._bg_features.get(cluster_id, {})

    def get_uid_booking_summary(self, uid):
        return self._uid_booking_summaries.loc[uid] if uid in self._uid_booking_summaries else {}

    @staticmethod
    def load(config):
        bdf = pd.read_csv(config['BOOKING_FEATURE_FILE_PATH'])
        bg_features = get_group_features(config['BG_FILE_PATH'])
        return BookingDataProvider(bdf, bg_features)


class ItemDataProvider(object):
    def __init__(self, pdf, bg_iids):
        self._prepare_active_iids(pdf)
        self._prepare_bg_iid_data(bg_iids)

    def _prepare_active_iids(self, pdf):
        self._active_iids = set(pdf[pdf.active == -1].propcode)

    def _prepare_bg_iid_data(self, bg_iids):
        self.iid_to_col = {}

        rows = []
        cols = []
        for bg_id, iids in bg_iids.items():
            rows += [bg_id] * len(iids)
            for iid in iids:
                cols.append(self.iid_to_col.setdefault(iid, len(self.iid_to_col)))

        self.bg_iid_m = csr_matrix((np.ones(len(rows)), (rows, cols)))

    def get_score_per_iid_row(self, iids, scores=None):
        if scores is not None:
            assert len(iids) == len(scores)

            cols = []
            data = []
            for iid, score in zip(iids, scores):
                if iid in self.iid_to_col:
                    cols.append(self.iid_to_col[iid])
                    data.append(score)
        else:
            cols = [self.iid_to_col[iid] for iid in iids if iid in self.iid_to_col]
            data = np.ones(len(cols))
        return csr_matrix((data, (np.zeros(len(cols)), cols)), shape=(1, len(self.iid_to_col)))

    def get_iid_per_bg_row(self, iid_mask, min_iid_per_bg):
        iid_per_bg = self.bg_iid_m.multiply(iid_mask).sum(axis=1).A1
        iid_per_bg[iid_per_bg < min_iid_per_bg] = 0
        return csr_matrix(iid_per_bg, shape=(1, iid_per_bg.size))

    def get_active_iids(self):
        return self._active_iids

    def prepare_bg_recs(self, bg_recs, iid_recs, top_clusters=None, top_items=None):
        _bg_iid_m = self.bg_iid_m.multiply(iid_recs)

        recs = {}
        for b_arg_id in np.argsort(bg_recs.data)[::-1][:top_clusters]:
            bg_id, bg_score = bg_recs.indices[b_arg_id], bg_recs.data[b_arg_id]
            _iid_recs = _bg_iid_m[bg_id]

            i_recs = {}
            for i_arg_id in np.argsort(_iid_recs.data)[::-1][:top_items]:
                iid, iid_score = _iid_recs.indices[i_arg_id], _iid_recs.data[i_arg_id]
                i_recs[iid] = iid_score

            recs[bg_id] = {
                "score": bg_score,
                "properties": i_recs
            }
        return recs

    @staticmethod
    def load(config):
        cols = ["propcode", "active"]
        pdf = pd.read_csv(config['PROPERTY_FILE_PATH'])[cols]
        bid_to_bgs, bg_iids = get_bg_data(config['BG_FILE_PATH'])
        return ItemDataProvider(pdf, bg_iids)
