import numba as nb
import numpy as np
from scipy.sparse import rand
import scipy.sparse as sp
from time import process_time as p_time


@nb.njit()
def power_iter(x, data, indptr, indices, p):
    for i in range(10):
        xlast = x
        mat_mul = vsmul_csr(x, data, indptr, indices)
        x = (1 - alpha) * mat_mul + alpha * p
        err = np.absolute(x - xlast).sum()
        if err <= 1e-10:
            break
    return x


def power_iter_2(x, A, p):
    for i in range(10):
        xlast = x
        mat_mul = x @ A
        x = (1 - alpha) * mat_mul + alpha * p
        err = np.absolute(x - xlast).sum()
        if err <= 1e-10:
            break
    return x


@nb.njit
def vsmul_csr(x, A, iA, jA):
    res = np.zeros(x.shape[0], dtype=np.float32)
    for row in nb.prange(len(iA) - 1):
        for i in nb.prange(iA[row], iA[row + 1]):
            data = A[i]
            row_i = row
            col_j = jA[i]
            res[col_j] += data * x[row_i]
    return res


if __name__ == "__main__":
    max_iter = 100
    alpha = 0.85
    n = 100
    p = np.zeros(n, dtype=float)
    p[0] = 0.5

    x = np.zeros(n, dtype=float)
    x[0] = 0.5

    A = rand(n, n, density=0.74, format="csr", random_state=42)
    S = np.array(A.sum(axis=1), dtype=np.float32).flatten()
    S[S != 0.0] = 1.0 / S[S != 0.0]
    Q = sp.spdiags(S.T, 0, *A.shape, format="csr")
    A = Q * A
    # print(A.todense().sum(axis=1))

    # res_1 = vsmul_csr(x, A.data, A.indptr, A.indices)
    # res_2 = x @ A
    # print(np.abs(res_1 - res_2).sum())

    time_arr = []
    for _ in range(10):
        _t_start = p_time()
        res_1 = power_iter(x, A.data, A.indptr, A.indices, p)
        res_2 = power_iter_2(x, A, p)
        print("error", np.abs(res_1 - res_2).sum())
        time_arr.append(p_time() - _t_start)
    time_rcrd = np.array(time_arr)
    print(np.mean(time_rcrd), np.std(time_rcrd))
