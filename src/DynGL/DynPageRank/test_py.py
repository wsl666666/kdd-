import numpy as np
from numba import njit
from numpy.linalg import norm
import networkx as nx


@njit(cache=True)
def var_appr_fista_vec(n, indptr, indices, degree, s, alpha, eps, rho, momentum_fixed, opt_x):
    """
    ---
    The goal of var_appr_* methods is to solve:
    min g(x) =  f(x) + alpha * rho * ||D^{1/2}x||_1,
    where
        f(y) = (1/2) y^TQ*y - alpha*(D^{-1/2}s)^T y
    and
        Q = D^{-1/2}(D-(1-alpha)(D+A)/2)D^{-1/2}
    using NAG algorithm

    :param n: the number of nodes of G
    :param indptr: csr_mat's indptr
    :param indices: csr_mat's indices
    :param degree: degree of G
    :param s: the initial vector (assume ||s||_1 = 1, s >= 0)
    :param alpha: dumping factor
    :param eps: precision factor
    :param rho: the SOR parameter
    :param momentum_fixed: True to use a fixed momentum parameter.
                            dynamically adjust parameter
    :param opt_x:
    :return:
    """
    # approximated solution
    xt = np.zeros(n, dtype=np.float64)
    xt_pre = np.zeros_like(xt)
    yt = np.zeros_like(xt)
    mat_yt_vec = np.zeros_like(xt)
    grad_yt = np.zeros(n, dtype=np.float64)

    # initialize to avoid redundant calculation
    sqrt_deg = np.zeros(n, dtype=np.float64)
    eps_vec = np.zeros(n, dtype=np.float64)
    # calculate grad of xt = 0 and S0
    # this part is has large run time O(n)
    for u in np.arange(n):
        sqrt_deg[u] = np.sqrt(degree[u])
        eps_vec[u] = rho * alpha * sqrt_deg[u]
        # calculate active nodes for first epoch
        if s[u] > 0.:
            grad_yt[u] = -alpha * s[u] / sqrt_deg[u]
    epoch_num = 0
    l1_error = []
    num_opers = []
    # parameter for momentum
    t1 = 1
    beta = (1. - np.sqrt(alpha)) / (1. + np.sqrt(alpha))
    num_oper = 0
    while True:
        for u in np.arange(n):
            mat_yt_vec[u] = 0.
            for v in indices[indptr[u]:indptr[u + 1]]:
                demon = sqrt_deg[v] * sqrt_deg[u]
                mat_yt_vec[u] += yt[v] / demon
        delta_yt = .5 * (1. - alpha) * (yt + mat_yt_vec) + alpha * s / sqrt_deg
        xt = np.sign(delta_yt) * np.maximum(np.abs(delta_yt) - eps_vec, 0.)
        grad_yt = yt - delta_yt
        if momentum_fixed:
            yt = xt + beta * (xt - xt_pre)
        else:
            t_next = .5 * (1. + np.sqrt(4. + t1 ** 2.))
            beta = (t1 - 1.) / t_next
            yt = xt + beta * (xt - xt_pre)
            t1 = t_next
        xt_pre[:] = xt
        if opt_x is not None:
            l1_error.append(norm(xt - opt_x, 1))
        else:
            l1_error.append(norm(xt - xt_pre, 1))
        num_oper += indptr[-1]
        num_opers.append(num_oper)
        epoch_num += 1
        # gradient is small enough
        cond = np.max(np.abs(grad_yt / sqrt_deg))
        if cond <= (1. + eps) * rho * alpha:
            break
    return xt, l1_error, num_opers


# n, indptr, indices, degree, s, alpha, eps, rho, opt_x
@njit(cache=True)
def var_appr_fista_queue(n, indptr, indices, degree, s, alpha, eps, rho, momentum_fixed, opt_x):
    """
    ---
    The goal of var_appr_* methods is to solve:
    min g(x) =  f(x) + alpha * rho * ||D^{1/2}x||_1,
    where
        f(y) = (1/2) y^TQ*y - alpha*(D^{-1/2}s)^T y
    and
        Q = D^{-1/2}(D-(1-alpha)(D+A)/2)D^{-1/2}
    using FISTA algorithm

    :param n: the number of nodes of G
    :param indptr: csr_mat's indptr
    :param indices: csr_mat's indices
    :param degree: degree of G
    :param s: the initial vector (assume ||s||_1 = 1, s >= 0)
    :param alpha: dumping factor
    :param eps: precision factor
    :param rho: the SOR parameter
    :param momentum_fixed: True to use a fixed momentum parameter.
                            dynamically adjust parameter
    :param opt_x:
    :return:
    """
    # queue to maintain active nodes per-epoch
    queue = np.zeros(n, dtype=np.int64)
    q_mark = np.zeros(n, dtype=np.bool_)
    q_len = np.int64(0)
    # approximated solution
    qt = np.zeros(n, dtype=np.float64)
    qt_pre = np.zeros_like(qt)
    yt = np.zeros(n, dtype=np.float64)
    grad_yt = np.zeros(n, dtype=np.float64)
    # initialize to avoid redundant calculation
    sqrt_deg = np.zeros(n, dtype=np.float64)
    eps_vec = np.zeros(n, dtype=np.float64)
    for i in range(n):
        sqrt_deg[i] = np.sqrt(degree[i])
        eps_vec[i] = rho * alpha * np.sqrt(degree[i])
        # calculate active nodes for first epoch
        if s[i] > 0.:
            grad_yt[i] = -alpha * s[i] / sqrt_deg[i]
        if (qt[i] - grad_yt[i]) > eps_vec[i] * (1. + eps):
            queue[q_len] = i
            q_len += 1
            q_mark[i] = True
    l1_error = []
    num_opers = []
    num_oper = 0
    if q_len == 0:
        # don't do anything
        return qt, l1_error, num_opers
    # parameter for momentum
    t1 = 1
    beta = (1. - np.sqrt(alpha)) / (1. + np.sqrt(alpha))
    while True:
        for ind in range(q_len):
            q_mark[queue[ind]] = False
        rear = 0
        for ind in range(q_len):
            i = queue[ind]
            if (yt[i] - grad_yt[i]) > eps_vec[i]:
                delta_qi = yt[i] - grad_yt[i] - eps_vec[i] - qt[i]
            elif (yt[i] - grad_yt[i]) < - eps_vec[i]:
                delta_qi = yt[i] - grad_yt[i] + eps_vec[i] - qt[i]
            else:
                delta_qi = -qt[i]
            qt[i] += delta_qi
            if momentum_fixed:
                delta_yi = qt[i] + beta * delta_qi - yt[i]
            else:
                t_next = .5 * (1. + np.sqrt(4. + t1 ** 2.))
                beta = (t1 - 1.) / t_next
                delta_yi = qt[i] + beta * delta_qi - yt[i]
                t1 = t_next

            yt[i] += delta_yi
            grad_yt[i] += .5 * (1. + alpha) * delta_yi
            for j in indices[indptr[i]:indptr[i + 1]]:
                demon = sqrt_deg[j] * sqrt_deg[i]
                ratio = .5 * (1 - alpha) / demon
                grad_yt[j] += (- ratio * delta_yi)
                if not q_mark[j] and np.abs(grad_yt[j]) > eps_vec[j] * (1. + eps):
                    queue[rear] = j
                    rear += 1
                    q_mark[j] = True
            num_oper += degree[i]
        qt_pre[:] = qt
        q_len = rear
        num_opers.append(num_oper)
        if opt_x is not None:
            l1_error.append(np.linalg.norm(qt - opt_x, 1))
        else:
            l1_error.append(np.linalg.norm(qt_pre - qt, 1))
        if q_len == 0:
            break
    return qt, l1_error, num_opers


@njit(cache=True)
def var_local_fista(n, indptr, indices, degree, s, alpha, eps, rho, momentum_fixed, opt_x):
    """
    ---
    The goal of var_appr_* methods is to solve:
    min g(x) =  f(x) + alpha * rho * ||D^{1/2}x||_1,
    where f(y) = (1/2) y^TQ*y - alpha*(D^{-1/2}s)^T y
    and Q = D^{-1/2}(D-(1-alpha)(D+A)/2)D^{-1/2}
    using FISTA algorithm

    :param n: the number of nodes of G
    :param indptr: csr_mat's indptr
    :param indices: csr_mat's indices
    :param degree: degree of G
    :param s: the initial vector (assume ||s||_1 = 1, s >= 0)
    :param alpha: dumping factor
    :param eps: precision factor
    :param rho: the SOR parameter
    :param momentum_fixed: True to use a fixed momentum parameter.
                            dynamically adjust parameter
    :param opt_x:
    :return:
    """
    # queue to maintain active nodes per-epoch
    queue = np.zeros(n, dtype=np.int64)
    q_mark = np.zeros(n, dtype=np.bool_)
    rear = np.int64(0)
    # approximated solution
    xt = np.zeros(n, dtype=np.float64)
    xt_pre = np.zeros_like(xt)
    yt = np.zeros_like(xt)
    grad_yt = np.zeros(n, dtype=np.float64)
    delta_yt = np.zeros_like(grad_yt)

    # initialize to avoid redundant calculation
    sq_deg = np.zeros(n, dtype=np.float64)
    eps_vec = np.zeros(n, dtype=np.float64)
    # calculate grad of xt = 0 and S0
    # this part is has large run time O(n)
    for u in np.arange(n):
        sq_deg[u] = np.sqrt(degree[u])
        eps_vec[u] = rho * alpha * sq_deg[u]
        # calculate active nodes for first epoch
        if s[u] > 0.:
            grad_yt[u] = -alpha * s[u] / sq_deg[u]
            if (xt[u] - grad_yt[u]) > eps_vec[u]:
                queue[rear] = u
                rear = rear + 1
                q_mark[u] = True
    epoch_num = 0
    errs = []
    opers = []
    if rear <= 0:  # empty active sets, don't do anything
        return xt, errs, opers
    # parameter for momentum
    t1 = 1
    beta = (1. - np.sqrt(alpha)) / (1. + np.sqrt(alpha))

    while True:
        st = queue[:rear]
        for u in st:
            delta_yt[u] = .5 * (1. - alpha) * yt[u] + alpha * s[u] / sq_deg[u]
        num_oper = 0.
        for u in st:
            num_oper += degree[u]
            for v in indices[indptr[u]:indptr[u + 1]]:
                demon = sq_deg[v] * sq_deg[u]
                delta_yt[v] += .5 * (1. - alpha) * yt[u] / demon
                # new active nodes added into st
                if not q_mark[v]:
                    queue[rear] = v
                    rear = rear + 1
                    q_mark[v] = True
        st = queue[:rear]
        xt[st] = np.sign(delta_yt[st]) * np.maximum(np.abs(delta_yt[st]) - eps_vec[st], 0.)
        grad_yt[st] = yt[st] - delta_yt[st]
        if momentum_fixed:
            yt[st] = xt[st] + beta * (xt[st] - xt_pre[st])
        else:
            t_next = .5 * (1. + np.sqrt(4. + t1 ** 2.))
            beta = (t1 - 1.) / t_next
            yt[st] = xt[st] + beta * (xt[st] - xt_pre[st])
            t1 = t_next
        xt_pre[st] = xt[st]
        if opt_x is not None:
            errs.append(norm(sq_deg * xt - sq_deg * opt_x, 1))
        else:
            errs.append(0.)
        opers.append(num_oper)
        epoch_num += 1
        # gradient is small enough
        cond = np.max(np.abs(grad_yt / sq_deg))
        if cond <= (1. + eps) * rho * alpha:
            break
    return sq_deg * xt, errs, opers



@njit(cache=True)
def _var_appr_fista_v3(n, indptr, indices, degree, s, alpha, eps, rho , momentum_fixed, opt_x=None):
    # approximated solution
    xt = np.zeros(n, dtype=np.float64)
    xt_pre = np.zeros_like(xt)
    yt = np.zeros_like(xt)
    mat_yt_vec = np.zeros_like(xt)
    # initialize to avoid redundant calculation
    sqrt_deg = np.zeros(n, dtype=np.float64)
    eps_vec = np.zeros(n, dtype=np.float64)
    for i in range(n):
        sqrt_deg[i] = np.sqrt(degree[i])
        eps_vec[i] = rho * alpha * np.sqrt(degree[i])
        # calculate active nodes for first epoch
    # parameter for momentum
    t1 = 1
    beta = (1. - np.sqrt(alpha)) / (1. + np.sqrt(alpha))

    gap_list = []
    nonzero_list = []
    nonzero_posi_list = []
    nonzero_nega_list = []

    touched_nodes = np.zeros(n, dtype=np.float64)
    optimal_nodes = np.zeros(n, dtype=np.float64)
    l1_error = []
    while True:
        for ii in np.arange(n):
            mat_yt_vec[ii] = 0.
            for jj in indices[indptr[ii]:indptr[ii + 1]]:
                demon = sqrt_deg[jj] * sqrt_deg[ii]
                mat_yt_vec[ii] += yt[jj] / demon
        delta_yt = .5 * (1. - alpha) * (yt + mat_yt_vec) + alpha * s / sqrt_deg
        xt = np.sign(delta_yt) * np.maximum(np.abs(delta_yt) - eps_vec, 0.)

        touched_nodes[np.nonzero(xt)[0]] = 1

        pr_vec = sqrt_deg * xt
        pr_vec_pre = sqrt_deg * xt_pre
        gap = np.linalg.norm(pr_vec - pr_vec_pre, 1)
        nonzero = np.count_nonzero(xt)
        nonzero_posi = np.count_nonzero(xt > 0.)
        nonzero_nega = np.count_nonzero(xt < 0.)
        gap_list.append(gap)
        nonzero_list.append(nonzero)
        nonzero_posi_list.append(nonzero_posi)
        nonzero_nega_list.append(nonzero_nega)
        if gap < eps * 1e-4:
            break
        if momentum_fixed:
            yt = xt + beta * (xt - xt_pre)
        else:
            t_next = .5 * (1. + np.sqrt(4. + t1 ** 2.))
            beta = (t1 - 1.) / t_next
            yt = xt + beta * (xt - xt_pre)
            t1 = t_next

        xt_pre = xt

        # if opt_x is not None:
        #     pr_vec = sqrt_deg * xt
        #     l1_err = np.linalg.norm(pr_vec - opt_x, 1)
        #     l1_error.append(l1_err)

    optimal_nodes[np.nonzero(xt)[0]] = 1
    expand_opt_nodes = np.zeros(n, dtype=np.float64)
    for ii in np.nonzero(xt)[0]:
        expand_opt_nodes[ii] = 1
        for jj in indices[indptr[ii]:indptr[ii + 1]]:
            expand_opt_nodes[jj] = 1
    total = 0
    for ii in np.nonzero(touched_nodes)[0]:
        if expand_opt_nodes[ii] == 0:
            total += 1
    return pr_vec, l1_error, nonzero_list, nonzero_posi_list, nonzero_nega_list


def main():
    import scipy.sparse as sp
    # from algo.ista import var_appr_ista
    # root = '/mnt/data2/baojian/git/appr-code/'
    # adj_m = sp.load_npz(root + 'datasets/com-dblp/com-dblp_csr-mat.npz')
    
    max_node_num = 3
    adj_m = sp.csr_matrix(
        (
            [1, 1, 1, 1,1,1],
            (
                [0, 1, 1, 2, 0, 2],
                [1, 0, 2, 1, 2, 0],
            ),
        ),
        shape=(max_node_num, max_node_num),
    )


    degree = adj_m.sum(1).A.flatten()
    seed_node = 0
    indices = adj_m.indices
    indptr = adj_m.indptr
    n = len(degree)
    s = np.zeros(n)
    s[seed_node] = 1.
    rho = 1e-6 / n
    alpha = 0.2
    eps = 1e-1
    # xt1, errs, opers = var_appr_ista(
    #     n, indptr, indices, degree, s, alpha, eps, rho, opt_x=None)
    g = nx.from_scipy_sparse_array(adj_m, create_using=nx.MultiGraph)
    x = nx.pagerank(g, alpha=1-alpha, personalization={seed_node : 1.0})
    print('nx ppr:', x)
    
    alpha = alpha/(2.0-alpha)
    #var_appr_fista_vec var_appr_fista_queue var_local_fista
    xt2, errs, opers = var_local_fista(
        n, indptr, indices, degree, s, alpha, eps, rho, momentum_fixed=False, opt_x=None)
    print('appr:', xt2)
    
    
    

if __name__ == '__main__':
    main()
