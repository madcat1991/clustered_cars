import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from misc.common import get_ug_data, get_bg_data, get_group_features


FEATURE_THRESHOLD = 0.5


class ObjFeatureSparseData(object):
    def __init__(self, m, obj_to_row, feature_to_col):
        self.m = m

        self.obj_to_row = obj_to_row
        self.feature_to_col = feature_to_col

        self.row_to_obj = {row_id: obj_id for obj_id, row_id in obj_to_row.items()}
        self.col_to_feature = {col_id: fid for fid, col_id in feature_to_col.items()}

    @property
    def n_objs(self):
        return len(self.obj_to_row)

    @property
    def n_features(self):
        return len(self.feature_to_col)

    def info(self):
        return {
            "nnz": self.m.nnz,
            "density": self.m.nnz / (self.m.shape[0] * self.m.shape[1]),
            "shape": self.m.shape
        }

    def get_obj_vector(self, obj_id):
        return self.m[self.obj_to_row[obj_id]]

    def get_objs_matrix(self, objs_ids):
        row_ids = [self.obj_to_row[obj_id] for obj_id in objs_ids]
        return self.m[row_ids]


class UserDataProvider(object):
    def __init__(self, udf, uid_to_ug, ug_features):
        # TODO probably later *df-s should be removed from constructors
        self._uid_to_ug = uid_to_ug
        self._ug_features = ug_features
        self._prepare_uid_features(udf)

    def _prepare_uid_features(self, udf):
        udf = udf.set_index("code")
        booking_cnt = udf.booking_cnt
        udf = udf.drop(["booking_cnt"], axis=1).apply(lambda x: x / booking_cnt)
        self._uid_features = udf

    def get_cluster_id(self, uid):
        return self._uid_to_ug.get(uid)

    def get_uid_features(self, uid):
        uid_features = {}
        if uid in self._uid_features.index:
            for feature, score in self._uid_features.loc[uid].items():
                if score > FEATURE_THRESHOLD:
                    uid_features[feature] = score
        return uid_features

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

        self._prepare_obs_per_iid(bdf)
        self._prepare_booking_iid_uid_part(bdf)
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
        self._uid_booking_summaries = data

    def get_iids_for_uid(self, uid):
        return set(self._b_iid_uid[self._b_iid_uid.code == uid].propcode)

    def get_obs_per_iid(self):
        return self._obs_per_iid

    def get_cluster_features(self, cluster_id):
        return self._bg_features.get(cluster_id, {})

    def get_uid_booking_summary(self, uid):
        summary = {}
        if uid in self._uid_booking_summaries.index:
            for feature, score in self._uid_booking_summaries.loc[uid].items():
                if score > FEATURE_THRESHOLD:
                    summary[feature] = score
        return summary

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

        self.col_to_iid = {col: iid for iid, col in self.iid_to_col.items()}
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

    def prepare_iid_recs(self, iid_recs, top_items=None):
        recs = []

        if top_items is None:
            arg_ids = np.argsort(iid_recs.data)[::-1]
        else:
            arg_ids = np.argsort(iid_recs.data)[-top_items:][::-1]

        for arg_id in arg_ids:
            recs.append({
                "propcode": self.col_to_iid[iid_recs.indices[arg_id]],
                "score": iid_recs.data[arg_id]
            })
        return recs

    def prepare_bg_recs(self, bg_recs, iid_recs, top_clusters=None, top_items=None):
        recs = []

        if top_clusters is None:
            arg_ids = np.argsort(bg_recs.data)[::-1]
        else:
            arg_ids = np.argsort(bg_recs.data)[-top_clusters:][::-1]

        _bg_iid_m = self.bg_iid_m.multiply(iid_recs)
        for arg_id in arg_ids:
            bg_id, bg_score = bg_recs.indices[arg_id], bg_recs.data[arg_id]
            recs.append({
                "bg_id": bg_id,
                "score": bg_score,
                "properties": self.prepare_iid_recs(_bg_iid_m[bg_id], top_items)
            })
        return recs

    @staticmethod
    def load(config):
        cols = ["propcode", "active"]
        pdf = pd.read_csv(config['PROPERTY_FILE_PATH'])[cols]
        bid_to_bgs, bg_iids = get_bg_data(config['BG_FILE_PATH'])
        return ItemDataProvider(pdf, bg_iids)


class ItemFeatureDataProvider(object):
    def __init__(self, bdf, pfdf):
        # obj-feature data based on items
        self._prepare_item_feature_data(pfdf)
        self._prepare_user_feature_data(bdf, pfdf)

    def _prepare_item_feature_data(self, pfdf):
        feature_to_col = {
            fid: col_id for col_id, fid in
            enumerate(pfdf.columns.drop(["year", "propcode"]))
        }
        data = pfdf.drop(["year"], axis=1).groupby("propcode").mean()
        iid_to_row = {iid: row_id for row_id, iid in enumerate(data.index)}
        m = csr_matrix(data.values, shape=(len(iid_to_row), len(feature_to_col)))
        self._ifd = ObjFeatureSparseData(m, iid_to_row, feature_to_col)

    def _prepare_user_feature_data(self, bdf, pfdf):
        _df = pd.merge(bdf, pfdf, on=["propcode", "year"])
        feature_to_col = {
            fid: col_id for col_id, fid in
            enumerate(_df.columns.drop(["propcode", "year", "code"]))
        }
        data = _df.drop(["propcode", "year"], axis=1).groupby("code").mean()
        uid_to_row = {uid: row_id for row_id, uid in enumerate(data.index)}
        m = csr_matrix(data.values, shape=(len(uid_to_row), len(feature_to_col)))
        self._ufd = ObjFeatureSparseData(m, uid_to_row, feature_to_col)

    def has_uid_features(self, uid):
        return uid in self._ufd.obj_to_row

    def get_uids_feature_matrix(self, uids):
        row_ids = [self._ufd.obj_to_row[uid] for uid in uids]
        return self._ufd.m[row_ids]

    def get_iids_feature_matrix(self, iids):
        row_ids = [self._ifd.obj_to_row[iid] for iid in iids]
        return self._ifd.m[row_ids]

    @staticmethod
    def load(config):
        cols = ["code", "propcode", "year"]
        bdf = pd.read_csv(config['BOOKING_FEATURE_FILE_PATH'])[cols]
        pfdf = pd.read_csv(config['PROPERTY_FEATURE_FILE_PATH'])
        return ItemFeatureDataProvider(bdf, pfdf)
