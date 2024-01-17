import numpy as np
from numba.typed import Dict as nb_dict
from numba.core import types
import numba as nb
from numpy.linalg import norm
import scipy as sp
import sys
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import Dict, List, Any

MAX_EPSILON_PRECISION = np.float64(1e-7)
MIN_EPSILON_PRECISION = np.float64(1e-15)


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
    update_ops_ppr_update: List[Dict]  # algorithm ppr update ops nums


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
            key_type=types.uint64, value_type=types.float64[:]
        )
        self.dict_r_arr = nb_dict.empty(
            key_type=types.uint64, value_type=types.float64[:]
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
                dtype=np.float64,
            )

        return None

    def _init_dict_r(
        self,
        init_node_ids: np.ndarray,
    ):
        """init typed dict of r, keyed by tracked node id.
        r[i] = 1.0 if i==u, else 0.0.

        """
        if self.ppr_algo == "ista":
            # alpha_lazy = np.float64(self.alpha / (2.0 - self.alpha))
            # for _u in init_node_ids:
            #     self.dict_r_arr[_u] = np.zeros(
            #         self.max_node_num,
            #         dtype=np.float64,
            #     )
            #     self.dict_r_arr[_u][_u] = -alpha_lazy
            for _u in init_node_ids:
                self.dict_r_arr[_u] = np.zeros(
                    self.max_node_num,
                    dtype=np.float64,
                )
                self.dict_r_arr[_u][_u] = 1.0
        else:
            for _u in init_node_ids:
                self.dict_r_arr[_u] = np.zeros(
                    self.max_node_num,
                    dtype=np.float64,
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
        elif self.ppr_algo == "fista":
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

        new_nodes_uni = np.unique(new_nodes.astype(np.uint64))
        should_add_node_ids = []
        for node_id in new_nodes_uni:
            node_id = np.uint64(node_id)
            if node_id not in self.track_nodes_set:
                should_add_node_ids.append(node_id)
                self.track_nodes_set.add(node_id)
        new_tracked_node_ids = np.array(should_add_node_ids).astype(np.uint64)
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
        snapshot_id: int,
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
        # ops
        ppr_updates_metric = {
            "power_iter_ops": np.uint64(0),
            "local_push_pos_push": np.uint64(0),
            "local_push_neg_push": np.uint64(0),
            "ista_ops": np.uint64(0),
        }

        if self.ppr_algo == "forward_push":
            # push specific params:
            beta = 0.0 if "beta" not in kwargs else kwargs["beta"]
            assert "init_epsilon" in kwargs, "init_epsilon?"
            init_epsilon = kwargs["init_epsilon"]

            # init_epsilon = (
            #     1e-6
            #     if "init_epsilon" not in kwargs
            #     else kwargs["init_epsilon"]
            # )

            # print(self.dict_p_arr.keys())
            # print(self.dict_r_arr.keys())
            (_, local_push_pos_push, local_push_neg_push, _,) = forward_push_routine(
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
            ppr_updates_metric["local_push_pos_push"] = local_push_pos_push
            ppr_updates_metric["local_push_neg_push"] = local_push_neg_push

        elif self.ppr_algo == "power_iteration":
            beta = 0.0 if "beta" not in kwargs else kwargs["beta"]
            assert "init_epsilon" in kwargs, "init_epsilon?"
            assert "power_iteration_max_iter" in kwargs, "max_iter?"
            init_epsilon = kwargs["init_epsilon"]
            max_iter = kwargs["power_iteration_max_iter"]
            # init_epsilon = (
            #     1e-6
            #     if "init_epsilon" not in kwargs
            #     else kwargs["init_epsilon"]
            # )
            # max_iter: int = (
            #     5000
            #     if "power_iteration_max_iter" not in kwargs
            #     else kwargs["power_iteration_max_iter"]
            # )
            (power_iter_ops, _, _, _,) = power_iteration_routine(
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
            ppr_updates_metric["power_iter_ops"] = power_iter_ops

        elif self.ppr_algo == "ista":
            # assert "ista_rho" in kwargs, "ISTA: ista_rho is not assigned!"
            # assert "ista_max_iter" in kwargs, "ISTA: ista_max_iter miss!"
            # assert "ista_erly_brk_tol" in kwargs, "ISTA: early exit-tol miss!"
            init_epsilon = kwargs["init_epsilon"]

            (_, _, _, ista_ops) = ista_ppr_routine(
                self.max_node_num,
                self.track_nodes,
                csr_indices,
                csr_indptr,
                csr_data,
                out_degree,
                self.dict_p_arr,
                self.dict_r_arr,  # no r-vec anymore
                alpha_norm=alpha,  # alpha of no-lazy random walk
                max_iter=kwargs["ista_max_iter"],
                init_epsilon=init_epsilon,
            )
            ppr_updates_metric["ista_ops"] = ista_ops
        elif self.ppr_algo == "fista":
            init_epsilon = kwargs["init_epsilon"]
            (_, _, _, fista_ops) = fista_ppr_local_routine(
                self.max_node_num,
                self.track_nodes,
                csr_indices,
                csr_indptr,
                csr_data,
                out_degree,
                self.dict_p_arr,
                self.dict_r_arr,
                alpha_norm=alpha,  # alpha of no-lazy random walk
                max_iter=-1, # ignore it
                init_epsilon=init_epsilon,
            )
            ppr_updates_metric["fista_ops"] = fista_ops
        
        else:
            raise NotImplementedError

        return ppr_updates_metric

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

            # crt_intermediate_degree_in = (
            #     cache_before_update["_in_degree"]
            #     + cache_before_update["_delta_in_degree"],
            # )

            # crt_intermediate_degree_out = (
            #     cache_before_update["_out_degree"]
            #     + cache_before_update["_delta_out_degree"]
            # )

            crt_intermediate_degree_in = cache_before_update[
                "_crt_intermediate_in_degree"
            ]
            crt_intermediate_degree_out = cache_before_update[
                "_crt_intermediate_out_degree"
            ]

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
        if self.ppr_algo in ("forward_push", "ista"):
            # if self.ppr_algo in ("forward_push"):
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
        elif self.ppr_algo == "fista":
            # TODO: add dynamic update for fista local ppr sor
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
        s = np.uint64(query_list[i])
        p_s = dict_p_arr[s]
        r_s = dict_r_arr[s]
        # Init residual to be 1.0 for further pushing: r_s[s] = 1.0
        #   or self.dict_r_arr[s][s] = 1.0

        # if s not in r_s:  # if this source node was never seen before.
        #     r_s[s] = np.float64(1.0)  # r_s = 1
        #     p_s[s] = np.float64(0.0)  # p_s = 0
        #     delta_s_r[s] += np.float64(1.0)  # the changed volumn.

        d_u = node_degree[e_u]
        if p_s[e_u] != np.float64(0.0) and (d_u - e_delta_w) != 0:
            # if(d_u - e_delta_w) != 0:
            p_s_u = np.float64(p_s[e_u] * d_u / (d_u - e_delta_w))
            p_s[e_u] = p_s_u
            delta_r_u = e_delta_w * p_s_u / (d_u * alpha)
            delta_r_v = (1 - alpha) * delta_r_u
            # delta_r_v = (1-alpha)*w*p[u]/(D[u]*alpha)
            r_s_u = r_s[e_u]
            # (p[u]/(D[u] = prev_p[u]/D[u]-1)
            r_s[e_u] = np.float64(r_s_u - delta_r_u)
            r_s[e_v] = np.float64(r_s[e_v] + delta_r_v)
        else:
            # print(
            #     "met new nodes",
            #     "\tp_s[e_u]:",
            #     p_s[e_u],
            #     "\td_u",
            #     d_u,
            #     "\te_delta_w",
            #     e_delta_w,
            #     "\td_u - e_delta_w:",
            #     d_u - e_delta_w,
            # )
            pass


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
    # eps_prime = np.float64(init_epsilon / node_degree.sum())
    epsilon = np.float64(init_epsilon / node_degree.sum())

    total_pos_ops = np.zeros(query_list.shape[0], dtype=np.uint64)
    total_neg_ops = np.zeros(query_list.shape[0], dtype=np.uint64)

    for i in nb.prange(query_list.shape[0]):
        # Multi-thread if using numba's nonpython mode. No GIL
        # Adaptive push according to p_s
        s = query_list[i]

        p_s: np.ndarray = dict_p_arr[s]
        r_s: np.ndarray = dict_r_arr[s]

        # 1: Positive FIFO Queue
        # r_s[v] > epsilon*node_degree[v]
        q_pos = nb.typed.List()
        q_pos_ptr = np.uint64(0)
        # NOTE: Numba Bug: element in set() never returns!!
        # SEE: https://github.com/numba/numba/issues/6543
        q_pos_marker = nb.typed.Dict.empty(
            key_type=nb.types.uint64,
            value_type=nb.types.boolean,
        )

        # scan all residual in r_s, select init for pushing
        # Or only maintain the top p_s[v] nodes for pushing? since it's
        # likely affect top-ppr

        for v in nb.prange(r_s.shape[0]):
            # v = v
            if r_s[v] > epsilon * node_degree[v]:
                q_pos.append(v)
                q_pos_marker[v] = True

        # Positive: pushing pushing!
        while np.uint64(len(q_pos)) > q_pos_ptr:
            u = q_pos[q_pos_ptr]
            q_pos_ptr += np.uint64(1)
            q_pos_marker.pop(u)
            deg_u = node_degree[u]
            r_s_u = r_s[u]

            if r_s_u > epsilon * deg_u:  # for positive
                p_s[u] += alpha * r_s_u
                push_residual = np.float64((1 - alpha) * r_s_u / deg_u)
                _v = indices[indptr[u] : indptr[u + 1]]
                _w = data[indptr[u] : indptr[u + 1]]
                for _ in range(_v.shape[0]):
                    total_pos_ops[i] += np.uint64(1)
                    v = _v[_]
                    w_u_v = np.float64(_w[_])
                    # should multply edge weights.
                    r_s[v] += np.float64(push_residual * w_u_v)
                    if v not in q_pos_marker:
                        q_pos.append(v)
                        q_pos_marker[v] = True
                # r_s[u] = (1-alpha)*r_s[u]*beta # beta=0 --> r_s[u] = 0
                r_s[u] = np.float64(0.0)

        # Add dummy +=0 trick to avoid numba bug when convert while into for.
        # SEE: https://github.com/numba/numba/issues/5156
        q_pos_ptr += np.uint64(0)

        # 2: Negative FIFO Queue
        # r_s[v] < -epsilon*node_degree[v]
        q_pos = nb.typed.List()
        q_pos_ptr = np.uint64(0)
        q_pos_marker = nb.typed.Dict.empty(
            key_type=nb.types.uint64,
            value_type=nb.types.boolean,
        )

        # scan all residual in r_s, select init for pushing
        for v in nb.prange(r_s.shape[0]):
            # v = v
            if r_s[v] < -epsilon * node_degree[v]:  # for negative
                q_pos.append(v)
                q_pos_marker[v] = True

        # Negative: pushing pushing!
        while np.uint64(len(q_pos)) > q_pos_ptr:
            u = q_pos[q_pos_ptr]
            q_pos_ptr += np.uint64(1)
            q_pos_marker.pop(u)
            deg_u = node_degree[u]
            r_s_u = r_s[u]
            if r_s_u < -epsilon * deg_u:  # for negative
                p_s[u] += alpha * r_s_u
                push_residual = np.float64((1 - alpha) * r_s_u / deg_u)
                _v = indices[indptr[u] : indptr[u + 1]]
                _w = data[indptr[u] : indptr[u + 1]]
                for _ in range(_v.shape[0]):
                    total_neg_ops[i] += np.uint64(1)
                    v = _v[_]
                    w_u_v = np.float64(_w[_])
                    # should multply edge weights.
                    r_s[v] += np.float64(push_residual * w_u_v)
                    if v not in q_pos_marker:
                        q_pos.append(v)
                        q_pos_marker[v] = True
                # r_s[u] = (1-alpha)*r_s[u]*beta # beta=0 --> r_s[u] = 0
                r_s[u] = np.float64(0.0)
        # Add dummy +=0 trick to avoid numba bug when convert while into for.
        # SEE: https://github.com/numba/numba/issues/5156
        q_pos_ptr += np.uint64(0)

    return [
        np.uint64(0),
        np.sum(total_pos_ops),
        np.sum(total_neg_ops),
        np.uint64(0),
    ]


@nb.njit(cache=True, parallel=False, fastmath=True, nogil=True)
def vsmul_csr(x, A, iA, jA):
    power_iter_ops = np.uint64(0)
    res = np.zeros(x.shape[0], dtype=np.float64)
    for row in nb.prange(len(iA) - 1):
        for i in nb.prange(iA[row], iA[row + 1]):
            data = A[i]
            row_i = row
            col_j = jA[i]
            res[col_j] += np.float64(data * x[row_i])
            power_iter_ops += 1
    return res.astype(np.float64), power_iter_ops


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
    num_nodes = np.uint64(query_list.shape[0])
    power_iter_ops = np.uint64(0)
    for i in nb.prange(num_nodes):
        s = query_list[i]
        power_iter_ops_local = np.uint64(0)
        # initial vector
        x: np.ndarray = dict_p_arr[s]
        if x.sum() != 0.0:
            x /= x.sum()
        # Personalization vector
        p = np.zeros(np.int64(N), dtype=np.float64)
        p[s] = np.float64(1.0)
        # power iteration: make up to max_iter iterations
        for _i in range(max_iter):
            xlast = x
            res, __power_iter_ops = vsmul_csr(x, data, indptr, indices)
            power_iter_ops_local += __power_iter_ops
            x = ((1.0 - alpha) * res + alpha * p).astype(np.float64)
            # check convergence, l1 norm
            err = np.absolute(x - xlast).sum()
            # print(err)
            if err < tol:
                dict_p_arr[s] = x.astype(np.float64)
                power_iter_ops += power_iter_ops_local
                break
    return np.uint64(power_iter_ops)


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
    # nodelist = np.arange(N, dtype=np.uint64)
    # dangling = None

    A = sp.sparse.csr_matrix((data, indices, indptr), shape=(N, N))
    S = np.array(A.sum(axis=1), dtype=np.float64).flatten()
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

    power_iter_ops = __power_iter_numba(
        np.uint64(N),
        np.uint64(query_list),
        A.indices,
        A.indptr,
        A.data,
        dict_p_arr,
        np.float64(alpha),
        np.uint64(max_iter),
        np.float64(tol),
    )

    return power_iter_ops, np.uint64(0), np.uint64(0), np.uint64(0)


# global gradient
@nb.njit(cache=True, parallel=True, fastmath=True, nogil=True)
def ista_ppr_routine(
    N: int,
    query_list: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    data: np.ndarray,
    node_degree: np.ndarray,
    dict_p_arr: nb_dict,
    dict_r_arr: nb_dict,
    alpha_norm: np.float64,
    max_iter: int,
    init_epsilon: np.float64 = np.float64(1e-7),
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
    total_ista_ops = np.zeros(len(query_list), dtype=np.uint64)

    alpha = alpha_norm
    eta = np.float64(1.0 / (2.0 - alpha))
    # eta = np.float64(1.0)
    # D^{-1/2}
    sqrt_d_out = np.sqrt(node_degree).astype(np.float64)
    nsqrt_d_out = np.zeros_like(sqrt_d_out).astype(np.float64)
    for i in nb.prange(N):
        _sqrt_d = sqrt_d_out[i]
        if _sqrt_d > 0.0:
            nsqrt_d_out[i] = np.float64(1.0) / _sqrt_d

    # optimality condition
    # init_epsilon = init_epsilon * 1e-3
    opt_cond_val = np.float64(init_epsilon / node_degree.sum()) * sqrt_d_out

    for s_i in nb.prange(len(query_list)):
        s_id = query_list[s_i]
        ista_ops = np.uint64(0)
        e_s = np.zeros(N, dtype=np.float64)
        e_s[s_id] = 1.0
        prev_ppv = np.zeros(N, dtype=np.float64)

        p = dict_p_arr[s_id]  # x = np.zeros(N, dtype=np.float64)
        x = np.multiply(nsqrt_d_out, p)
        # x = np.zeros(N, dtype=np.float64)

        # r = dict_r_arr[s_id]  # grad_f_q = np.zeros(N, dtype=np.float64)
        # grad_f = np.multiply(nsqrt_d_out, r)  # working
        grad_f = np.zeros(N, dtype=np.float64)
        z = np.zeros(N, dtype=np.float64)

        for iter_num in range(max_iter):
            # eval gradient now Wx+b
            for i in range(N):
                dAd = np.float64(0.0)
                for ptr in range(indptr[i], indptr[i + 1]):
                    j = indices[ptr]
                    dAd += nsqrt_d_out[i] * data[ptr] * nsqrt_d_out[j] * x[j]
                grad_f[i] = x[i] - (1 - alpha) * dAd - alpha * nsqrt_d_out[i] * e_s[i]
                z[i] = x[i] - eta * grad_f[i]

            # solve proximal
            for i in range(N):
                if z[i] > opt_cond_val[i]:
                    ista_ops += np.uint64(1)
                    x[i] = z[i] - opt_cond_val[i]
                elif np.abs(z[i]) < opt_cond_val[i]:
                    x[i] = np.float64(0)
                elif z[i] < -opt_cond_val[i]:
                    ista_ops += np.uint64(1)
                    x[i] = z[i] + opt_cond_val[i]
                else:
                    pass

            ppv = np.multiply(sqrt_d_out, x)
            if (
                np.absolute(ppv - prev_ppv).sum() < 0.01 * init_epsilon
                # np.multiply(grad_f, sqrt_d_out).sum() < init_epsilon
                or iter_num == max_iter - 1
            ):
                dict_p_arr[s_id] = ppv
                dict_r_arr[s_id] = np.multiply(grad_f, sqrt_d_out)
                total_ista_ops[s_i] += ista_ops
                break
            prev_ppv = ppv

    return (np.uint64(0), np.uint64(0), np.uint64(0), np.sum(total_ista_ops))



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
    nodelist = np.arange(N, dtype=np.uint64)
    dangling = None

    A = sp.sparse.csr_matrix((data, indices, indptr), shape=(N, N))
    S = np.array(A.sum(axis=1), dtype=np.float64).flatten()
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
        personalization = np.zeros(N, dtype=np.float64)
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
                dict_p_arr[s] = x.astype(np.float64)
                break


@nb.njit(cache=True, parallel=True, fastmath=True, nogil=True)
def fista_ppr_local_routine(
    N: int,
    query_list: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    data: np.ndarray,
    node_degree: np.ndarray,
    dict_p_arr: nb_dict,
    dict_r_arr: nb_dict,
    alpha_norm: np.float64,
    max_iter: int,
    init_epsilon: np.float64 = np.float64(1e-5),
    rho:np.float64 = np.float64(1e-5),
    mome_fixed: bool = False,
):
    """wrapper of var_local_fista to calculate PPR vector in parallel"""
    
    # convert between lazy walk and non-lazy walk alpha.
    alpha_norm =  alpha_norm/(2.0-alpha_norm)
    total_fista_ops = np.zeros(len(query_list), dtype=np.uint64)
    for s_i in nb.prange(len(query_list)):
        opers = var_local_fista(N, indptr, indices, node_degree, dict_p_arr, dict_r_arr, query_list[s_i], alpha_norm, init_epsilon, rho, mome_fixed)
        total_fista_ops[s_i] = opers
    
    return (np.uint64(0), np.uint64(0), np.uint64(0), np.sum(total_fista_ops))


@nb.njit(cache=True, parallel=True, fastmath=True, nogil=True)
def var_local_fista(n, indptr, indices, degree, dict_p_arr, dict_r_arr, s_i, alpha, eps, rho, mome_fixed):
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
    :param dict_p_arr: the numba dict to record final ppv
    :param dict_r_arr: the numba dict to record final residual
    :param query_list: the numba dict to record final residual
    :param s_i: the node index
    :param alpha: dumping factor
    :param eps: precision factor
    :param rho: the SOR parameter
    :param mome_fixed: True to use a fixed momentum parameter.
                            dynamically adjust parameter
    :return:
    """
    # the initial vector (assume ||s||_1 = 1, s >= 0)
    # s = dict_p_arr[s_i] 
    # init here
    s = np.zeros(n, dtype = np.float64)
    s[s_i] = 1.0
    
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
    epoch_num = np.uint32(0)
    # errs = []
    opers = 0
    if rear <= 0:  # empty active sets, don't do anything
        dict_p_arr[s_i] = xt
        return np.uint32(0)
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
        if mome_fixed:
            yt[st] = xt[st] + beta * (xt[st] - xt_pre[st])
        else:
            t_next = .5 * (1. + np.sqrt(4. + t1 ** 2.))
            beta = (t1 - 1.) / t_next
            yt[st] = xt[st] + beta * (xt[st] - xt_pre[st])
            t1 = t_next
        xt_pre[st] = xt[st]
        # if opt_x is not None:
        #     errs.append(norm(sq_deg * xt - sq_deg * opt_x, 1))
        # else:
        #     errs.append(0.)
        opers+=num_oper
        epoch_num += 1
        # gradient is small enough
        cond = np.max(np.abs(grad_yt / sq_deg))
        if cond <= (1. + eps) * rho * alpha:
            break
    # record final result
    ppv = sq_deg * xt
    dict_p_arr[s_i] = ppv
    # TODO: check the residual conversion
    dict_r_arr[s_i] = np.abs(grad_yt / sq_deg) #np.multiply(grad_f, sqrt_d_out)

    return opers