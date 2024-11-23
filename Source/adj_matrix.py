import numpy as np


ADJ_MATRIX = np.array([
    [0, 1, 0, 1, 1],
    [1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1],
    [1, 0, 0, 1, 0],
])


if __name__ == '__main__':
    id = 1
    neighbours = (ADJ_MATRIX[id - 1]).nonzero()[0] + 1
    print(neighbours)
