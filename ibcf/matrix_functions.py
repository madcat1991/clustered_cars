import numpy as np
from scipy.sparse import csr_matrix


def get_topk(matrix, top, axis=1):
    """Converts source matrix to Top-K matrix
    where each row or column contains only top K values

    :param matrix: source matrix
    :param top: number of top items to be stored
    :param axis: 0 - top by column, 1 - top by row
    :return:
    """
    rows = []
    cols = []
    data = []

    if axis == 0:
        matrix = matrix.T.tocsr()

    for row_id, row in enumerate(matrix):
        if top is not None and row.nnz > top:
            top_args = np.argsort(row.data)[-top:]

            rows += [row_id] * top
            cols += row.indices[top_args].tolist()
            data += row.data[top_args].tolist()
        elif row.nnz > 0:
            rows += [row_id] * row.nnz
            cols += row.indices.tolist()
            data += row.data.tolist()

    topk_m = csr_matrix((data, (rows, cols)), (matrix.shape[0], matrix.shape[1]))

    if axis == 0:
        topk_m = topk_m.T.tocsr()

    return topk_m


def get_top_from_row(row, top=None):
    """Get top elements from a row

    :param row: sparse row
    :param top: number of element that we assume to retrieve
    """
    if top is not None and row.nnz > top:
        top_args = np.argsort(row.data)[-top:]

        rows = [0] * top
        cols = row.indices[top_args].tolist()
        data = row.data[top_args].tolist()

        row = csr_matrix((data, (rows, cols)), shape=row.shape)
    return row


def nullify_main_diagonal(m):
    positions = range(m.shape[0])
    eye = csr_matrix((np.ones(len(positions)), (positions, positions)), m.shape)
    m = m - m.multiply(eye)
    return m


def get_sparse_matrix_info(m):
    return {
        "shape": m.shape,
        "nnz": m.nnz,
        "density": round(float(m.nnz) / (m.shape[0] * m.shape[1]), 4) if m.shape != (0, 0) else 0
    }
