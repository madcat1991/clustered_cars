# coding: utf-8

from sklearn.preprocessing import normalize
from matrix_functions import nullify_main_diagonal, get_topk


def get_ib_topk_cosine_sim(ui_matrix, top=None):
    """
    The method builds item-item cosine similarity matrix
    according to: http://dl.acm.org/citation.cfm?id=963776

    :param ui_matrix: user x item matrix
    :param top: number of similarities per object in the final matrix
    """
    user_norm_m = normalize(ui_matrix, axis=1)
    iu_norm_m = normalize(user_norm_m.T.tocsr(), axis=1)

    sim_m = iu_norm_m.dot(iu_norm_m.T)
    sim_m = nullify_main_diagonal(sim_m)

    if top is not None:
        sim_m = get_topk(sim_m, top, axis=0)
    # be aware of this normalization since it plays very special role during
    # final recommendations calculation
    return normalize(sim_m, axis=0)  # column-wise


def get_similarity_matrix(ui_matrix, top=None):
    sim_m = get_ib_topk_cosine_sim(ui_matrix, top)
    return sim_m.tocsr()
