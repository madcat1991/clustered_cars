"""
Methods for item-based top-k CF recommendations
"""

from ibcf.matrix_functions import get_topk


def get_topk_recs(ui_vector, sim_matrix, e_matrix=None, top=None):
    """
    Getting top recommendations for user with some history

    :param ui_vector: user x item vector
    :param sim_matrix: similarity matrix
    :param e_matrix: user-item pairs that should be excluded from recs (binary matrix)
    :param top: number of recommendations
    :return:
    """
    # this order of dot production is obligatory
    recs_row = ui_vector.dot(sim_matrix)
    if e_matrix is not None:
        recs_row = recs_row - recs_row.multiply(e_matrix)
    return get_topk(recs_row, top) if top is not None else recs_row
