import matplotlib.pyplot as plt
import numpy as np


# ADJ_MATRIX = np.array([
#     [0, 1, 0, 1, 1],
#     [1, 0, 1, 0, 0],
#     [0, 1, 0, 1, 0],
#     [1, 0, 1, 0, 1],
#     [1, 0, 0, 1, 0],
# ])


ADJ_MATRIX = np.array([
    [0,   0.95,  0,    1,    1],
    [0.91, 0,    1,    0,    0],
    [0,   1,    0,    0.87, 0],
    [1,   0,    0.97, 0,    1],
    [1,   0,    0,    1,    0],
])

# ADJ_MATRIX = np.array([[0, 1, 0, 0, 0], [1, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 0]])
#
# ADJ_MATRIX = np.array([
#     [0, 1, 1, 0, 1],
#     [1, 0, 0, 0, 0],
#     [1, 0, 0, 1, 1],
#     [0, 0, 1, 0, 1],
#     [1, 0, 1, 1, 0]
# ])


if __name__ == '__main__':
    # from scipy.sparse.csgraph import laplacian
    # from scipy.sparse import diags
    # from scipy import linalg


    A = np.array([
        [0, 0, 0, 1, 1],
        [1, 0, 0, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
    ])

    # a = np.array([[0., -1.], [1., 0.]])
    # linalg.eigvals(a)

    # assert np.all(np.transpose(ADJ_MATRIX) == ADJ_MATRIX)
    # id = 1
    # neighbours = (ADJ_MATRIX[id - 1]).nonzero()[0] + 1
    # print(neighbours)
    d = np.sum(A, axis=1)
    print(d)
    L = A - diags(d)
    print(L)

    ev = linalg.eigvals(L)
    print(ev)
    ev_real_abs = np.abs(np.real(ev))

    plt.plot(np.real(ev_real_abs), '*')
    plt.show()
