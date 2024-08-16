from sklearn.preprocessing import normalize
from ibcf.matrix_functions import nullify_main_diagonal, get_topk


def get_ib_topk_cosine_sim(ui_matrix, top=None):
    """
    The method builds item-item cosine similarity matrix
    according to: http://dl.acm.org/citation.cfm?id=963776

    :param ui_matrix: user x item matrix
    :param top: number of similarities per object in the final matrix
    """
    iu_m = ui_matrix.T.tocsr()
    iu_norm_m = normalize(iu_m, axis=1)

    sim_m = iu_norm_m.dot(iu_norm_m.T)
    sim_m = nullify_main_diagonal(sim_m)

    if top is not None:
        sim_m = get_topk(sim_m, top, axis=0)

    return sim_m


def get_similarity_matrix(ui_matrix, top=None):
    return get_ib_topk_cosine_sim(ui_matrix, top)
