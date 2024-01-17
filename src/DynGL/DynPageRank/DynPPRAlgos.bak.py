import numpy as np
from numba.typed import Dict as nb_dict
from numba.core import types
import numba as nb
import scipy as sp
import sys
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import Dict, List, Any

MAX_EPSILON_PRECISION = np.float32(1e-7)
MIN_EPSILON_PRECISION = np.float32(1e-15)


@dataclass_json
@dataclass
class PPVResult:
    ppr_algo: str
    ppr_alpha: float
    ppr_algo_param_dict: Dict[str, Any]
    # ppr_arr_per_snapshot: Dict[int, Any]  # per node and t
    ppr_arr_snapshot_tmp_file: Dict[int, str]  # offload to disk path
    ppr_nnz_per_snapshot: Dict[int, np.ndarray]  # per node and t
    ppr_l1_norm_per_snapshot: Dict[int, np.ndarray]  # per node and t
    ppr_l1_err_per_snapshot: Dict[int, np.ndarray]  # per node and t
    update_time_graph_update: List[float]  # data strcuture time
    update_time_ppr_update: List[float]  # algorithm ppr update time


@dataclass
class CSRGraph:
    indptr = np.ndarray
    indices = np.ndarray
    data = np.ndarray


class DynPPREstimator:
    """Incremental PPR Calulation"""

    def __init__(
        self,
        max_node_num: int,
        track_nodes: np.ndarray,  # uint32
        alpha: float = 0.15,
        ppr_algo: str = None,
        incrmt_ppr: bool = True,
    ):
        self.max_node_num = max_node_num
        self.track_nodes = np.unique(track_nodes)
        self.track_nodes_set = set(list(self.track_nodes.flatten()))

        self.dict_p_arr = nb_dict.empty(
            key_type=types.uint32, value_type=types.float32[:]
        )
        self.dict_r_arr = nb_dict.empty(
            key_type=types.uint32, value_type=types.float32[:]
        )
        self.ppr_algo = ppr_algo
        self.alpha = alpha

        self.incrmt_ppr: bool = incrmt_ppr

        self._init_dict_p(self.track_nodes)
        self._init_dict_r(self.track_nodes)

    def _init_dict_p(
        self,
        init_node_ids: np.ndarray,
    ):
        """init typed dict of p, keyed by tracked node id"""
        for _u in init_node_ids:
            self.dict_p_arr[_u] = np.zeros(
                self.max_node_num,
                dtype=np.float32,
            )

        return None

    def _init_dict_r(
        self,
        init_node_ids: np.ndarray,
    ):
        """init typed dict of r, keyed by tracked node id.
        r[i] = 1.0 if i==u, else 0.0.

        """
        for _u in init_node_ids:
            self.dict_r_arr[_u] = np.zeros(
                self.max_node_num,
                dtype=np.float32,
            )
            self.dict_r_arr[_u][_u] = 1.0
        return None

    def __str__(self):
        if self.ppr_algo == "forward_push":
            printable_str = (
                f"PPR Algorithm:\t{self.ppr_algo}\n"
                f"tracked nodes:\t{len(self.dict_p_arr)}\n"
                f"tracked pprs:\t{self.dict_p_arr}"
                f"incremental update:\t{self.incrmt_ppr}"
            )
        elif self.ppr_algo == "power_iteration":
            printable_str = (
                f"PPR Algorithm:\t{self.ppr_algo}\n"
                f"tracked nodes:\t{len(self.dict_p_arr)}\n"
                f"tracked pprs:\t{self.dict_p_arr}"
                f"incremental update:\t{self.incrmt_ppr}"
            )
        elif self.ppr_algo == "ista":
            printable_str = (
                f"PPR Algorithm:\t{self.ppr_algo}\n"
                f"tracked nodes:\t{len(self.dict_p_arr)}\n"
                f"tracked pprs:\t{self.dict_p_arr}"
                f"incremental update:\t{self.incrmt_ppr}"
            )

        else:
            raise NotImplementedError
        return printable_str

    def add_nodes_to_ppr_track(
        self,
        new_nodes: np.ndarray,
    ):
        """dynamically add tracked nodes to the ppr estimator"""

        if new_nodes.shape[0] == 0:
            return None

        new_nodes_uni = np.unique(new_nodes.astype(np.uint32))
        should_add_node_ids = []
        for node_id in new_nodes_uni:
            node_id = int(node_id)
            if node_id not in self.track_nodes_set:
                should_add_node_ids.append(node_id)
                self.track_nodes_set.add(node_id)
        new_tracked_node_ids = np.array(should_add_node_ids).astype(np.uint32)
        # print(
        #     f"add new {new_tracked_node_ids.shape[0]} nodes to"
        #     f" tracking{new_tracked_node_ids}"
        # )
        self._init_dict_p(new_tracked_node_ids)
        self._init_dict_r(new_tracked_node_ids)

        self.track_nodes = np.hstack((self.track_nodes, new_tracked_node_ids))

    def update_ppr(
        self,
        csr_indptr: np.ndarray,
        csr_indices: np.ndarray,
        csr_data: np.ndarray,
        out_degree: np.ndarray,
        in_degree: np.ndarray,
        alpha: float,
        *args,
        **kwargs,
    ):
        """incrementally update ppr given current sparse graph
        using method, which in-place modifies dict_p_arr):
        - Local push
        - Power Iteration
        - L1 regularization solvers (ista, others are not implemented)

        Args:
            csr_indptr (np.ndarray): csr matrix indptr array
            csr_indices (np.ndarray): csr matrix indices array
            csr_data (np.ndarray): csr matrix data array
            out_degree (np.ndarray): grpah out-degree array
            in_degree (np.ndarray): graph in-degree array
            alpha (float): teleport probability in PPR

        Raises:
            NotImplementedError
        """

        if self.ppr_algo == "forward_push":
            # push specific params:
            beta = 0.0 if "beta" not in kwargs else kwargs["beta"]
            init_epsilon = (
                1e-6
                if "init_epsilon" not in kwargs
                else kwargs["init_epsilon"]
            )

            forward_push_routine(
                self.max_node_num,
                self.track_nodes,
                csr_indices,
                csr_indptr,
                csr_data,
                out_degree,
                self.dict_p_arr,
                self.dict_r_arr,
                alpha,
                beta,
                init_epsilon,
            )

        elif self.ppr_algo == "power_iteration":
            beta = 0.0 if "beta" not in kwargs else kwargs["beta"]
            init_epsilon = (
                1e-6
                if "init_epsilon" not in kwargs
                else kwargs["init_epsilon"]
            )
            max_iter: int = (
                5000
                if "power_iteration_max_iter" not in kwargs
                else kwargs["power_iteration_max_iter"]
            )
            power_iteration_routine(
                self.max_node_num,
                self.track_nodes,
                csr_indices,
                csr_indptr,
                csr_data,
                out_degree,
                self.dict_p_arr,
                self.dict_r_arr,
                alpha,
                beta,
                init_epsilon,
                max_iter=max_iter,
            )

        elif self.ppr_algo == "ista":
            assert "ista_rho" in kwargs, "ISTA: ista_rho is not assigned!"
            assert "ista_max_iter" in kwargs, "ISTA: ista_max_iter miss!"
            assert "ista_erly_brk_tol" in kwargs, "ISTA: early exit-tol miss!"

            ista_ppr_routine(
                self.max_node_num,
                self.track_nodes,
                csr_indices,
                csr_indptr,
                csr_data,
                out_degree,
                self.dict_p_arr,
                None,  # no r-vec anymore
                alpha_norm=alpha,  # alpha of normal random walk
                rho=kwargs["ista_rho"],  # 1e-5, l1 regularization param
                max_iter=kwargs["ista_max_iter"],  # 100,
                early_break_tol=kwargs["ista_erly_brk_tol"],  # 1e-7
            )

        else:
            raise NotImplementedError

    def callback_handle_func_all_edge_struct_event_after(
        self,
        cache_before_update: Dict,
        update_kwargs: Dict,
    ):
        # callback function after processing a batch of edge struct
        # events
        if not self.incrmt_ppr:
            print(
                "Warning: re-init p/r due to self.incrmt_ppr ="
                f" {self.incrmt_ppr}."
                "\n\tPPR calculation is NOT incremental. "
                "Is it for benchmark purpose?"
            )
            self._init_dict_p(self.track_nodes)
            self._init_dict_r(self.track_nodes)
        else:
            if self.ppr_algo in ("power_iteration"):  # "ista"
                print(
                    "Warning: re-init p/r for power-iter "
                    "power-iter is very slow when re-use previous state"
                )
                self._init_dict_p(self.track_nodes)
                self._init_dict_r(self.track_nodes)
            else:
                pass

        return update_kwargs

    def callback_handle_func_single_edge_struct_event_after(
        self,
        src,
        tgt,
        delta_w_consolidate,
        e_iter,
        cache_before_update,
        update_kwargs,
    ):
        if self.incrmt_ppr:
            # incrementally adjust ppr per edge event
            # cache_before_update
            #   ["_delta_in_degree"]: delta in-degree of this
            #   edge in curt batch
            #   ["_in_degree"]: in-degree before any update
            #   for curt batch.
            crt_intermediate_degree_in = (
                cache_before_update["_in_degree"]
                + cache_before_update["_delta_in_degree"],
            )

            crt_intermediate_degree_out = (
                cache_before_update["_out_degree"]
                + cache_before_update["_delta_out_degree"]
            )

            self.dynamic_adjust_ppr_per_edge(
                src,
                tgt,
                delta_w_consolidate,
                crt_intermediate_degree_out,
                crt_intermediate_degree_in,
                self.alpha,
            )
        else:
            pass

    def dynamic_adjust_ppr_per_edge(
        self,
        e_u: int,
        e_v: int,
        e_delta_w: float,  # delta w_{(u,v)}
        out_degree: np.ndarray,
        in_degree: np.ndarray,
        alpha: float,
        *args,
        **kargs,
    ):
        if self.ppr_algo == "forward_push":
            dynamic_adjust_ppr_per_edge_numba(
                self.track_nodes,
                self.dict_p_arr,
                self.dict_r_arr,
                out_degree,
                e_u,
                e_v,
                e_delta_w,
                alpha,
            )
        elif self.ppr_algo == "power_iteration":
            pass
        elif self.ppr_algo == "ista":
            pass
        else:
            raise NotImplementedError


@nb.njit(cache=True, parallel=True, fastmath=True, nogil=True)
def dynamic_adjust_ppr_per_edge_numba(
    query_list: np.ndarray,
    dict_p_arr: nb_dict,
    dict_r_arr: nb_dict,
    node_degree: np.ndarray,
    e_u: int,
    e_v: int,
    e_delta_w: float,  # delta w_{(u,v)}
    alpha: float,
    # delta_s_r,
):
    # Adjust vectors of P_s and R_s for every s w.r.t, new edge
    # (u, v, w)
    # Extended from Zhang 2016 dynamic push adjustment, but with
    # weighted edges delta_r for each

    for i in nb.prange(query_list.shape[0]):  # for every source node.
        s = np.uint32(query_list[i])
        p_s = dict_p_arr[s]
        r_s = dict_r_arr[s]
        # Init residual to be 1.0 for further pushing: r_s[s] = 1.0
        #   or self.dict_r_arr[s][s] = 1.0

        # if s not in r_s:  # if this source node was never seen before.
        #     r_s[s] = np.float32(1.0)  # r_s = 1
        #     p_s[s] = np.float32(0.0)  # p_s = 0
        #     delta_s_r[s] += np.float32(1.0)  # the changed volumn.

        d_u = node_degree[e_u]
        if p_s[e_u] != np.float32(0.0) and (d_u - e_delta_w) != 0:
            p_s_u = np.float32(p_s[e_u] * d_u / (d_u - e_delta_w))
            p_s[e_u] = p_s_u
            delta_r_u = e_delta_w * p_s_u / (d_u * alpha)
            delta_r_v = (1 - alpha) * delta_r_u
            # delta_r_v = (1-alpha)*w*p[u]/(D[u]*alpha)
            r_s_u = r_s[e_u]
            # (p[u]/(D[u] = prev_p[u]/D[u]-1)
            r_s[e_u] = np.float32(r_s_u - delta_r_u)
            r_s[e_v] = np.float32(r_s[e_v] + delta_r_v)


@nb.njit(cache=True, parallel=True, fastmath=True, nogil=True)
def forward_push_routine(
    N: int,
    query_list: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    data: np.ndarray,
    node_degree: np.ndarray,
    dict_p_arr: nb_dict,
    dict_r_arr: nb_dict,
    alpha: float,
    beta: float,
    init_epsilon: float,
):
    """Calculate PPR based on Andersen's local push algorithm with
    Numba multi-thread acceleration.


    Args:
        N (int): total number of nodes |V|
        query_list (np.ndarray): the queried nodes for ppr
        indices (np.ndarray): indices arr for csr graph
        indptr (np.ndarray): indptr arr for csr graph
        data (np.ndarray): edge_w arr for csr graph
        node_degree (np.ndarray): node out-degree arr
        dict_p_arr (nb_dict): estimated PPR vector indexed by node-id
        dict_r_arr (nb_dict): estimated Residual vector ind by node-id
        alpha (float): Teleport probability in PPR vector
        beta (float): = 0,
        init_epsilon (float): for the first push.
    """
    # print("1. start push")
    # eps_prime = np.float32(init_epsilon / node_degree.sum())

    for i in nb.prange(query_list.shape[0]):
        # Multi-thread if using numba's nonpython mode. No GIL
        # Adaptive push according to p_s
        s = query_list[i]
        # for high degree nodes?
        # adapt_epsilon = np.float32(eps_prime * node_degree[s])
        # adapt_epsilon = np.max(
        #     np.array([adapt_epsilon, MIN_EPSILON_PRECISION])
        # )  # the lower bound
        # epsilon = np.min(
        #     np.array([adapt_epsilon, MAX_EPSILON_PRECISION])
        # )
        epsilon = np.float32(init_epsilon)
        # the upper bound
        # print("epsilon:", epsilon)

        p_s: np.ndarray = dict_p_arr[s]
        r_s: np.ndarray = dict_r_arr[s]

        # 1: Positive FIFO Queue
        # r_s[v] > epsilon*node_degree[v]
        q_pos = nb.typed.List()
        q_pos_ptr = np.uint64(0)
        # NOTE: Numba Bug: element in set() never returns!!
        # SEE: https://github.com/numba/numba/issues/6543
        q_pos_marker = nb.typed.Dict.empty(
            key_type=nb.types.uint32,
            value_type=nb.types.boolean,
        )

        # scan all residual in r_s, select init for pushing
        # Or only maintain the top p_s[v] nodes for pushing? since it's
        # likely affect top-ppr

        for v in nb.prange(r_s.shape[0]):
            v = np.uint32(v)
            if r_s[v] > epsilon * node_degree[v]:
                q_pos.append(v)
                q_pos_marker[v] = True

        # Positive: pushing pushing!
        num_pushes_pos = np.uint64(0)
        while np.uint64(len(q_pos)) > q_pos_ptr:
            u = q_pos[q_pos_ptr]
            q_pos_ptr += np.uint64(1)
            q_pos_marker.pop(u)
            deg_u = node_degree[u]
            r_s_u = r_s[u]

            if r_s_u > epsilon * deg_u:  # for positive
                num_pushes_pos += np.uint64(1)
                p_s[u] += alpha * r_s_u
                push_residual = np.float32((1 - alpha) * r_s_u / deg_u)
                _v = indices[indptr[u] : indptr[u + 1]]
                _w = data[indptr[u] : indptr[u + 1]]
                for _ in range(_v.shape[0]):
                    v = np.uint32(_v[_])
                    w_u_v = np.float32(_w[_])
                    # should multply edge weights.
                    r_s[v] += np.float32(push_residual * w_u_v)
                    if v not in q_pos_marker:
                        q_pos.append(v)
                        q_pos_marker[v] = True
                # r_s[u] = (1-alpha)*r_s[u]*beta # beta=0 --> r_s[u] = 0
                r_s[u] = np.float32(0.0)

        # Add dummy +=0 trick to avoid numba bug when convert while into for.
        # SEE: https://github.com/numba/numba/issues/5156
        q_pos_ptr += np.uint64(0)

        # 2: Negative FIFO Queue
        # r_s[v] < -epsilon*node_degree[v]
        q_pos = nb.typed.List()
        q_pos_ptr = np.uint64(0)
        q_pos_marker = nb.typed.Dict.empty(
            key_type=nb.types.uint32,
            value_type=nb.types.boolean,
        )

        # scan all residual in r_s, select init for pushing
        for v in nb.prange(r_s.shape[0]):
            v = np.uint32(v)
            if r_s[v] < -epsilon * node_degree[v]:  # for negative
                q_pos.append(v)
                q_pos_marker[v] = True

        num_pushes_neg = np.uint64(0)
        # Negative: pushing pushing!
        while np.uint64(len(q_pos)) > q_pos_ptr:
            u = q_pos[q_pos_ptr]
            q_pos_ptr += np.uint64(1)
            q_pos_marker.pop(u)
            deg_u = node_degree[u]
            r_s_u = r_s[u]
            if r_s_u < -epsilon * deg_u:  # for negative
                num_pushes_neg += 1
                p_s[u] += alpha * r_s_u
                push_residual = np.float32((1 - alpha) * r_s_u / deg_u)
                _v = indices[indptr[u] : indptr[u + 1]]
                _w = data[indptr[u] : indptr[u + 1]]
                for _ in range(_v.shape[0]):
                    v = np.uint32(_v[_])
                    w_u_v = np.float32(_w[_])
                    # should multply edge weights.
                    r_s[v] += np.float32(push_residual * w_u_v)
                    if v not in q_pos_marker:
                        q_pos.append(v)
                        q_pos_marker[v] = True
                # r_s[u] = (1-alpha)*r_s[u]*beta # beta=0 --> r_s[u] = 0
                r_s[u] = np.float32(0.0)
        # Add dummy +=0 trick to avoid numba bug when convert while into for.
        # SEE: https://github.com/numba/numba/issues/5156
        q_pos_ptr += np.uint64(0)


@nb.njit(cache=True, parallel=False, fastmath=True, nogil=True)
def vsmul_csr(x, A, iA, jA):
    res = np.zeros(x.shape[0], dtype=np.float32)
    for row in nb.prange(len(iA) - 1):
        for i in nb.prange(iA[row], iA[row + 1]):
            data = A[i]
            row_i = row
            col_j = jA[i]
            res[col_j] += np.float32(data * x[row_i])
    return res.astype(np.float32)


@nb.njit(cache=True, parallel=True, fastmath=True, nogil=True)
def __power_iter_numba(
    N,
    query_list,
    indices,
    indptr,
    data,
    dict_p_arr,
    alpha,
    max_iter,
    tol,
):
    num_nodes = np.uint32(query_list.shape[0])
    for i in nb.prange(num_nodes):
        s = query_list[i]
        # initial vector
        x: np.ndarray = dict_p_arr[s]
        if x.sum() != 0.0:
            x /= x.sum()
        # Personalization vector
        p = np.zeros(np.int64(N), dtype=np.float32)
        p[s] = np.float32(1.0)
        # power iteration: make up to max_iter iterations
        for _i in range(max_iter):
            xlast = x
            res = vsmul_csr(x, data, indptr, indices)
            x = ((1.0 - alpha) * res + alpha * p).astype(np.float32)
            # check convergence, l1 norm
            err = np.absolute(x - xlast).sum()
            # print(err)
            if err < tol:
                dict_p_arr[s] = x.astype(np.float32)
                break


def power_iteration_routine(
    N: int,
    query_list: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    data: np.ndarray,
    node_degree: np.ndarray,
    dict_p_arr: nb_dict,
    dict_r_arr: nb_dict,
    alpha: float,
    beta: float,
    init_epsilon: float,
    max_iter: int,
    *args,
    **kwargs,
):
    """Calculate PPR based on power iteration algorithm


    Args:
        N (int): total number of nodes |V|
        query_list (np.ndarray): the queried nodes for ppr
        indices (np.ndarray): indices arr for csr graph
        indptr (np.ndarray): indptr arr for csr graph
        data (np.ndarray): edge_w arr for csr graph
        node_degree (np.ndarray): node out-degree arr
        dict_p_arr (nb_dict): estimated PPR vector indexed by node-id
        dict_r_arr (nb_dict): estimated Residual vector ind by node-id
        alpha (float): Teleport probability in PPR vector
        beta (float): = 0,
        init_epsilon (float): early exit
    """

    tol: int = init_epsilon
    # nodelist = np.arange(N, dtype=np.uint32)
    # dangling = None

    A = sp.sparse.csr_matrix((data, indices, indptr), shape=(N, N))
    S = np.array(A.sum(axis=1), dtype=np.float32).flatten()
    S[S != 0.0] = 1.0 / S[S != 0.0]  # 1 over degree
    Q = sp.sparse.spdiags(S.T, 0, *A.shape, format="csr")
    A = Q * A
    # ensure no dangling nodes.
    if S[S == 0.0].shape[0] == 0:
        print(
            "warning: the graph has dangling node "
            "If the graph is undirected, it should be fine "
            "since the dangling node is isolated."
        )
    # assert S[S == 0.0].shape[0] == 0, "graph has dangling nodes"

    __power_iter_numba(
        np.uint32(N),
        np.uint32(query_list),
        A.indices,
        A.indptr,
        A.data,
        dict_p_arr,
        np.float32(alpha),
        np.uint32(max_iter),
        np.float32(tol),
    )


@nb.njit(cache=True, parallel=True, fastmath=True, nogil=True)
def ista_ppr_routine(
    # N: int,
    # s_id: int,
    # indices: np.ndarray,
    # indptr: np.ndarray,
    # data: np.ndarray,
    # node_degree: np.ndarray,
    # alpha_norm: np.float32,
    # rho: np.float32,
    # max_iter: int,
    # early_break_tol: np.float32 = np.float32(1e-7),
    N: int,
    query_list: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    data: np.ndarray,
    node_degree: np.ndarray,
    dict_p_arr: nb_dict,
    dict_r_arr: nb_dict,
    alpha_norm: np.float32,
    rho: np.float32,
    max_iter: int,
    early_break_tol: np.float32 = np.float32(1e-7),
):
    """calculate PPR vector using ISTA in the lens of L-1 regularized
    optimization.

    Args:
        N (int): total number of nodes |V|
        query_list (np.ndarray): the queried nodes for ppr
        indices (np.ndarray): indices arr for csr graph
        indptr (np.ndarray): indptr arr for csr graph
        data (np.ndarray): edge_w arr for csr graph
        node_degree (np.ndarray): node out-degree arr
        dict_p_arr (nb_dict): estimated PPR vector indexed by node-id
        dict_r_arr (nb_dict): estimated Residual vector ind by node-id
        alpha_norm (float): teleport proba of random walk (not lazy walk)
        rho (float): the l1 regularization param
        max_iter (int): the max iteration
        early_break_tol (float): the early exit tolerance of ppv
            (per node).
    """
    # the corresponding alpha of lazy random walk coverted from normal
    # random walk .
    alpha = np.float32(alpha_norm / (2.0 - alpha_norm))
    sqrt_d_out = np.sqrt(node_degree).astype(np.float32)
    nsqrt_d_out = (1.0 / sqrt_d_out).astype(np.float32)
    Q_data = np.zeros_like(indices, dtype=np.float32)
    con2 = np.float32(0.5 * (1.0 + alpha))
    # num_ops = 0

    # Q = D^{-1/2} @ A @ D^{-1/2}
    for u in nb.prange(N):
        _nsqrt_d_out = nsqrt_d_out[u]
        for ptr in nb.prange(indptr[u], indptr[u + 1]):
            Q_data[ptr] = _nsqrt_d_out * nsqrt_d_out[indices[ptr]] * data[ptr]
    Q_data = (0.5 * (alpha - 1.0) * Q_data).astype(np.float32)
    cond_vec = (rho * alpha * sqrt_d_out).astype(np.float32)

    for s_i in nb.prange(len(query_list)):
        s_id = query_list[s_i]
        q = np.zeros(N, dtype=np.float32)
        grad_f = np.zeros(N, dtype=np.float32)
        grad_f[s_id] = np.float32(-alpha * nsqrt_d_out[s_id])
        grad_c = alpha * nsqrt_d_out[s_id]
        ppv_prev = np.zeros(N, dtype=np.float32)

        for k in range(max_iter):
            # for i in nb.prange(N):
            for i in nb.prange(N):
                delta = q[i] - grad_f[i]
                if delta >= cond_vec[i]:
                    q[i] = delta - cond_vec[i]
                # elif concond_val_vec[i]d_val <= -cond_vec[i]:  # won't happen
                #     q[i] = q[i] - (grad_f[i] - cond_vec[i])
                else:
                    q[i] = np.float32(0.0)

            # if k > 200:
            ppv = np.multiply(sqrt_d_out, q)
            if np.absolute(ppv - ppv_prev).sum() < N * early_break_tol:
                dict_p_arr[s_id] = ppv
                break
            ppv_prev = ppv

            # get gradient. Qq - alpha*nsqrt_d*e_i
            grad_f = con2 * q
            grad_f[s_id] -= grad_c
            for u in nb.prange(N):
                _sum = np.float32(0.0)
                for ptr in range(indptr[u], indptr[u + 1]):
                    _sum += q[indices[ptr]] * Q_data[ptr]
                    # num_ops += 1
                grad_f[u] += _sum


@nb.njit(cache=True, parallel=True, fastmath=True, nogil=True)
def ista_ppr_routine_deprecated(
    N: int,
    query_list: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    data: np.ndarray,
    node_degree: np.ndarray,
    dict_p_arr: nb_dict,
    dict_r_arr: nb_dict,
    alpha_norm: np.float32,
    rho: np.float32,
    max_iter: int,
    early_break_tol: np.float32 = np.float32(1e-7),
):
    """calculate PPR vector using ISTA in the lens of L-1 regularized
    optimization.

    Args:
        N (int): total number of nodes |V|
        query_list (np.ndarray): the queried nodes for ppr
        indices (np.ndarray): indices arr for csr graph
        indptr (np.ndarray): indptr arr for csr graph
        data (np.ndarray): edge_w arr for csr graph
        node_degree (np.ndarray): node out-degree arr
        dict_p_arr (nb_dict): estimated PPR vector indexed by node-id
        dict_r_arr (nb_dict): estimated Residual vector ind by node-id
        alpha_norm (float): teleport proba of random walk (not lazy walk)
        rho (float): the l1 regularization param
        max_iter (int): the max iteration
        early_break_tol (float): the early exit tolerance of ppv
            (per node).
    """
    # the corresponding alpha of lazy random walk coverted from normal
    # random walk .
    alpha = alpha_norm / (2.0 - alpha_norm)
    sqrt_d_out = np.sqrt(node_degree).astype(np.float32)
    nsqrt_d_out = 1.0 / sqrt_d_out
    Q_data = np.zeros_like(data, dtype=np.float32)

    # Q = D^{-1/2} @ A @ D^{-1/2}
    for u in nb.prange(N):
        _ptr_start = indptr[u]
        _ptr_end = indptr[u + 1]
        for ptr in nb.prange(_ptr_start, _ptr_end):
            v = indices[ptr]
            Q_data[ptr] = nsqrt_d_out[u] * nsqrt_d_out[v] * data[ptr]
    Q_data = (0.5 * (alpha - 1) * Q_data).astype(np.float32)
    cond_vec = (rho * alpha * sqrt_d_out).astype(np.float32)

    for _s_id in nb.prange(query_list.shape[0]):
        s_id = query_list[_s_id]
        q = np.zeros(N, dtype=np.float32)
        grad_f = np.zeros(N, dtype=np.float32)
        grad_f[s_id] = np.float32(-alpha * nsqrt_d_out[s_id])
        Cc = np.zeros(N, dtype=np.float32)
        Cc[s_id] = alpha * nsqrt_d_out[s_id]

        ppv_prev = np.zeros(N, dtype=np.float32)

        for k in range(max_iter):
            cond_val_vec = q - grad_f
            for i in nb.prange(N):
                cond_val = cond_val_vec[i]
                if cond_val >= cond_vec[i]:
                    q[i] = q[i] - (grad_f[i] + cond_vec[i])
                elif cond_val <= -cond_vec[i]:  # won't happen
                    q[i] = q[i] - (grad_f[i] - cond_vec[i])
                else:
                    q[i] = np.float32(0.0)

            # get gradient. Qq - alpha*nsqrt_d*e_i
            # grad_f_2 = Q_2 @ q + 0.5 * (1 + alpha) * q - Cc
            grad_f = (0.5 * (1.0 + alpha) * q - Cc).astype(np.float32)
            for u in nb.prange(N):
                _ptr_start = indptr[u]
                _ptr_end = indptr[u + 1]
                for ptr in nb.prange(_ptr_start, _ptr_end):
                    v = indices[ptr]
                    d = Q_data[ptr]
                    grad_f[u] += np.float32(q[v] * d)

            ppv = np.zeros(N, dtype=np.float32)
            for i in nb.prange(N):
                ppv[i] = sqrt_d_out[i] * q[i]

            if np.absolute(ppv - ppv_prev).sum() < N * early_break_tol:
                break
            ppv_prev = ppv

        # save back
        dict_p_arr[s_id] = ppv


def ista_ppr_routine_python(
    N: int,
    query_list: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    data: np.ndarray,
    node_degree: np.ndarray,
    dict_p_arr: nb_dict,
    dict_r_arr: nb_dict,
    alpha_norm: float,
    rho: float,
    max_iter: int,
):
    """calculate PPR vector using ISTA in the lens of L-1 regularized
    optimization.

    Args:
        N (int): total number of nodes |V|
        query_list (np.ndarray): the queried nodes for ppr
        indices (np.ndarray): indices arr for csr graph
        indptr (np.ndarray): indptr arr for csr graph
        data (np.ndarray): edge_w arr for csr graph
        node_degree (np.ndarray): node out-degree arr
        dict_p_arr (nb_dict): estimated PPR vector indexed by node-id
        dict_r_arr (nb_dict): estimated Residual vector ind by node-id
        alpha_norm (float): teleport proba of random walk (not lazy walk)
        rho (float): the l1 regularization param
        max_iter (int): the max iteration
    """
    print(
        f"Warning: {sys._getframe().f_code.co_name} is implemented in Python. "
        "It produce ppr from scratch every time"
    )
    # lazy walk is equivalent to random walk with adjusted alpha.
    alpha = alpha_norm / (2.0 - alpha_norm)
    A = sp.sparse.csr_matrix((data, indices, indptr), shape=(N, N))
    d_mat = sp.sparse.diags(node_degree, dtype=np.float32)
    sqrt_d_out = np.sqrt(node_degree)
    nsqrt_d_out = 1.0 / sqrt_d_out
    nsqrt_d_out_mat = sp.sparse.diags(nsqrt_d_out, dtype=np.float32)
    # I = sp.sparse.identity(N, dtype=np.float32)
    Q = (
        nsqrt_d_out_mat
        @ (d_mat - 0.5 * (1 - alpha) * (d_mat + A))
        @ nsqrt_d_out_mat
    )
    cond_vec = rho * alpha * (sqrt_d_out)

    for s_id in query_list:
        q = np.zeros(N, dtype=np.float32)
        grad_f = np.zeros(N, dtype=np.float32)
        grad_f[s_id] = -alpha * (nsqrt_d_out[s_id])

        Cc = np.zeros(N, dtype=np.float32)
        Cc[s_id] = alpha * nsqrt_d_out[s_id]

        for k in range(max_iter):
            for i in range(N):
                if q[i] - grad_f[i] >= cond_vec[i]:
                    q[i] = q[i] - (grad_f[i] + cond_vec[i])
                elif q[i] - grad_f[i] <= -cond_vec[i]:
                    q[i] = q[i] - (grad_f[i] - cond_vec[i])
                else:
                    q[i] = 0.0
            # get gradient. Qq - alpha*nsqrt_d*e_i
            grad_f = Q @ q - Cc

        ppv = np.zeros(N, dtype=np.float32)
        for i in range(N):
            ppv[i] = sqrt_d_out[i] * q[i]
        dict_p_arr[s_id] = ppv


@nb.njit(cache=True, parallel=True, fastmath=True, nogil=True)
def ista_ppr_routine_dirty(
    N: int,
    query_list: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    data: np.ndarray,
    node_degree: np.ndarray,
    dict_p_arr: nb_dict,
    dict_r_arr: nb_dict,
    alpha_norm: np.float32,
    rho: np.float32,
    max_iter: int,
):
    """calculate PPR vector using ISTA in the lens of L-1 regularized
    optimization.

    Args:
        N (int): total number of nodes |V|
        query_list (np.ndarray): the queried nodes for ppr
        indices (np.ndarray): indices arr for csr graph
        indptr (np.ndarray): indptr arr for csr graph
        data (np.ndarray): edge_w arr for csr graph
        node_degree (np.ndarray): node out-degree arr
        dict_p_arr (nb_dict): estimated PPR vector indexed by node-id
        dict_r_arr (nb_dict): estimated Residual vector ind by node-id
        alpha_norm (float): teleport proba of random walk (not lazy walk)
        rho (float): the l1 regularization param
        max_iter (int): the max iteration
    """
    # print(
    #     f"Warning: {sys._getframe().f_code.co_name} is  in Python. "
    #     "It produce ppr from scratch every time"
    # )
    # lazy walk is equivalent to random walk with adjusted alpha.
    alpha = alpha_norm / (2.0 - alpha_norm)
    # A = sp.sparse.csr_matrix((data, indices, indptr), shape=(N, N))
    sqrt_d_out = np.sqrt(node_degree).astype(np.float32)
    nsqrt_d_out = 1.0 / sqrt_d_out
    # nsqrt_d_out_mat = sp.sparse.diags(nsqrt_d_out, dtype=np.float32)
    # Q_2 = -0.5 * (1 - alpha) * (nsqrt_d_out_mat @ A @ nsqrt_d_out_mat)
    Q_data = np.zeros_like(data, dtype=np.float32)
    for u in nb.prange(N):
        _ptr_start = indptr[u]
        _ptr_end = indptr[u + 1]
        for ptr in nb.prange(_ptr_start, _ptr_end):
            v = indices[ptr]
            Q_data[ptr] = nsqrt_d_out[u] * nsqrt_d_out[v] * data[ptr]
    Q_data = 0.5 * (alpha - 1) * Q_data
    Q_data = Q_data.astype(np.float32)
    # Q_2 = sp.sparse.csr_matrix((Q_data, indices, indptr), shape=(N, N))
    cond_vec = (rho * alpha * (sqrt_d_out)).astype(np.float32)

    for _s_id in nb.prange(query_list.shape[0]):
        s_id = query_list[_s_id]
        q = np.zeros(N, dtype=np.float32)
        grad_f = np.zeros(N, dtype=np.float32)
        grad_f[s_id] = np.float32(-alpha * nsqrt_d_out[s_id])

        Cc = np.zeros(N, dtype=np.float32)
        Cc[s_id] = alpha * nsqrt_d_out[s_id]

        for k in nb.prange(max_iter):
            cond_val_vec = q - grad_f
            for i in nb.prange(N):
                # print(type(q[i]), type(grad_f[i]))
                # cond_val = q[i] - grad_f[i]
                cond_val = cond_val_vec[i]
                if cond_val >= cond_vec[i]:
                    q[i] = q[i] - (grad_f[i] + cond_vec[i])
                elif cond_val <= -cond_vec[i]:
                    q[i] = q[i] - (grad_f[i] - cond_vec[i])
                else:
                    q[i] = np.float32(0.0)

            # get gradient. Qq - alpha*nsqrt_d*e_i
            # grad_f_2 = Q_2 @ q + 0.5 * (1 + alpha) * q - Cc
            grad_f = (0.5 * (1.0 + alpha) * q - Cc).astype(np.float32)
            for u in nb.prange(N):
                _ptr_start = indptr[u]
                _ptr_end = indptr[u + 1]
                for ptr in nb.prange(_ptr_start, _ptr_end):
                    v = indices[ptr]
                    d = Q_data[ptr]
                    grad_f[u] += np.float32(q[v] * d)

        ppv = np.zeros(N, dtype=np.float32)
        for i in nb.prange(N):
            ppv[i] = sqrt_d_out[i] * q[i]
        dict_p_arr[s_id] = ppv


@nb.njit(cache=True, parallel=True, fastmath=True, nogil=True)
def forward_push_routine_buggy(
    N: int,
    query_list: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    data: np.ndarray,
    node_degree: np.ndarray,
    dict_p_arr: nb_dict,
    dict_r_arr: nb_dict,
    alpha: float,
    beta: float,
    init_epsilon: float,
):
    """Calculate PPR based on Andersen's local push algorithm with
    Numba multi-thread acceleration.

    Note that: numba.set() is not stable!
        numb_set = numba.set()
        'v in numb_set' will hang
        # SEE: https://github.com/numba/numba/issues/6543
        It has not been solved as Dec 21, 2022


    Args:
        N (int): total number of nodes |V|
        query_list (np.ndarray): the queried nodes for ppr
        indices (np.ndarray): indices arr for csr graph
        indptr (np.ndarray): indptr arr for csr graph
        data (np.ndarray): edge_w arr for csr graph
        node_degree (np.ndarray): node out-degree arr
        dict_p_arr (nb_dict): estimated PPR vector indexed by node-id
        dict_r_arr (nb_dict): estimated Residual vector ind by node-id
        alpha (float): Teleport probability in PPR vector
        beta (float): = 0,
        init_epsilon (float): for the first push.
    """
    eps_prime = np.float32(init_epsilon / node_degree.sum())

    for i in nb.prange(query_list.shape[0]):
        # Multi-thread if using numba's nonpython mode. No GIL
        # Adaptive push according to p_s
        s = query_list[i]
        # for high degree nodes?
        adapt_epsilon = np.float32(eps_prime * node_degree[s])
        adapt_epsilon = np.max(
            np.array([adapt_epsilon, MIN_EPSILON_PRECISION])
        )  # the lower bound
        epsilon = np.min(
            np.array([adapt_epsilon, MAX_EPSILON_PRECISION])
        )  # the upper bound

        p_s: np.ndarray = dict_p_arr[s]
        r_s: np.ndarray = dict_r_arr[s]

        # 1: Positive FIFO Queue
        # r_s[v] > epsilon*node_degree[v]
        q_pos = nb.typed.List()
        q_pos_ptr = np.uint64(0)
        q_pos_marker_set = set()
        # NOTE: Numba Bug: element in set() never returns!!
        # SEE: https://github.com/numba/numba/issues/6543

        # scan all residual in r_s, select init for pushing
        # Or only maintain the top p_s[v] nodes for pushing? since it's
        # likely affect top-ppr

        for v in nb.prange(r_s.shape[0]):
            v = np.uint32(v)
            if r_s[v] > epsilon * node_degree[v]:
                q_pos.append(v)
                q_pos_marker_set.add(v)

        # Positive: pushing pushing!
        num_pushes_pos = np.uint64(0)
        while np.uint64(len(q_pos)) > q_pos_ptr:
            u = q_pos[q_pos_ptr]
            q_pos_ptr += np.uint64(1)

            q_pos_marker_set.remove(u)
            deg_u = node_degree[u]
            r_s_u = r_s[u]

            if r_s_u > epsilon * deg_u:  # for positive
                num_pushes_pos += np.uint64(1)
                p_s[u] += alpha * r_s_u
                push_residual = np.float32((1 - alpha) * r_s_u / deg_u)
                _v = indices[indptr[u] : indptr[u + 1]]
                _w = data[indptr[u] : indptr[u + 1]]
                for _ in range(_v.shape[0]):
                    v = np.uint32(_v[_])
                    w_u_v = np.float32(_w[_])
                    # should multply edge weights.
                    r_s[v] += np.float32(push_residual * w_u_v)
                    if v not in q_pos_marker_set:
                        q_pos.append(v)
                        q_pos_marker_set.add(v)
                # r_s[u] = (1-alpha)*r_s[u]*beta # beta=0 --> r_s[u] = 0
                r_s[u] = np.float32(0.0)

        # Add dummy +=0 trick to avoid numba bug when convert while into for.
        # SEE: https://github.com/numba/numba/issues/5156
        q_pos_ptr += np.uint64(0)

        # 2: Negative FIFO Queue
        # r_s[v] < -epsilon*node_degree[v]
        q_pos = nb.typed.List()
        q_pos_ptr = np.uint64(0)
        q_pos_marker_set = set()
        # scan all residual in r_s, select init for pushing
        for v in nb.prange(r_s.shape[0]):
            v = np.uint32(v)
            if r_s[v] < -epsilon * node_degree[v]:  # for negative
                q_pos.append(v)
                q_pos_marker_set.add(v)

        num_pushes_neg = np.uint64(0)
        # Negative: pushing pushing!
        while np.uint64(len(q_pos)) > q_pos_ptr:
            u = q_pos[q_pos_ptr]
            q_pos_ptr += np.uint64(1)
            q_pos_marker_set.remove(u)
            deg_u = node_degree[u]
            r_s_u = r_s[u]
            if r_s_u < -epsilon * deg_u:  # for negative
                num_pushes_neg += 1
                p_s[u] += alpha * r_s_u
                push_residual = np.float32((1 - alpha) * r_s_u / deg_u)
                _v = indices[indptr[u] : indptr[u + 1]]
                _w = data[indptr[u] : indptr[u + 1]]
                for _ in range(_v.shape[0]):
                    v = np.uint32(_v[_])
                    w_u_v = np.float32(_w[_])
                    # should multply edge weights.
                    r_s[v] += np.float32(push_residual * w_u_v)
                    if v not in q_pos_marker_set:
                        q_pos.append(v)
                        q_pos_marker_set.add(v)
                # r_s[u] = (1-alpha)*r_s[u]*beta # beta=0 --> r_s[u] = 0
                r_s[u] = np.float32(0.0)
        # Add dummy +=0 trick to avoid numba bug when convert while into for.
        # SEE: https://github.com/numba/numba/issues/5156
        q_pos_ptr += np.uint64(0)


def power_iteration_routine_python(
    N: int,
    query_list: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    data: np.ndarray,
    node_degree: np.ndarray,
    dict_p_arr: nb_dict,
    dict_r_arr: nb_dict,
    alpha: float,
    beta: float,
    init_epsilon: float,
    max_iter: int,
    *args,
    **kwargs,
):
    """Calculate PPR based on power iteration algorithm


    Args:
        N (int): total number of nodes |V|
        query_list (np.ndarray): the queried nodes for ppr
        indices (np.ndarray): indices arr for csr graph
        indptr (np.ndarray): indptr arr for csr graph
        data (np.ndarray): edge_w arr for csr graph
        node_degree (np.ndarray): node out-degree arr
        dict_p_arr (nb_dict): estimated PPR vector indexed by node-id
        dict_r_arr (nb_dict): estimated Residual vector ind by node-id
        alpha (float): Teleport probability in PPR vector
        beta (float): = 0,
        init_epsilon (float): early exit
    """

    tol: int = init_epsilon
    nodelist = np.arange(N, dtype=np.uint32)
    dangling = None

    A = sp.sparse.csr_matrix((data, indices, indptr), shape=(N, N))
    S = np.array(A.sum(axis=1), dtype=np.float32).flatten()
    S[S != 0.0] = 1.0 / S[S != 0.0]
    Q = sp.sparse.spdiags(S.T, 0, *A.shape, format="csr")
    A = Q * A

    for s in query_list:
        # initial vector
        nstart: np.ndarray = dict_p_arr[s]

        if np.sum(nstart) == 0.0:
            # For the first time
            x = np.repeat(1.0 / N, N)
        else:
            # x = np.array([nstart[_] for _ in nodelist], dtype=float)
            x = nstart
            x /= x.sum()

        # Personalization vector
        personalization = np.zeros(N, dtype=np.float32)
        personalization[s] = 1.0
        if personalization is None:
            p = np.repeat(1.0 / N, N)
        else:
            p = personalization
            if p.sum() == 0:
                raise ZeroDivisionError
            p /= p.sum()

        # Dangling nodes
        if dangling is None:
            dangling_weights = p
        else:
            # Convert the dangling dictionary into an array in nodelist order
            dangling_weights = np.array(
                [dangling.get(n, 0) for n in nodelist], dtype=float
            )
            dangling_weights /= dangling_weights.sum()
        is_dangling = np.where(S == 0)[0]

        # power iteration: make up to max_iter iterations
        for iter_num in range(max_iter):
            xlast = x
            x = (1 - alpha) * (
                x @ A + np.sum(x[is_dangling]) * dangling_weights
            ) + alpha * p
            # check convergence, l1 norm
            err = np.absolute(x - xlast).sum()
            if err < N * tol:
                # return dict(zip(nodelist, map(float, x)))
                dict_p_arr[s] = x.astype(np.float32)
                break
