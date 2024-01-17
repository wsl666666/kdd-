import unittest
import numpy as np
from DynPPRAlgos import DynPPREstimator
from scipy import sparse as sp


class test_dyngl(unittest.TestCase):
    def test_ppr_init(self):
        max_node_num = 3
        csr_graph = sp.csr_matrix(
            (
                [1, 1, 1, 1],
                (
                    [0, 1, 1, 2],
                    [1, 0, 2, 1],
                ),
            ),
            shape=(max_node_num, max_node_num),
        )

        out_degree = np.sum(csr_graph, axis=1, dtype=np.float32)
        out_degree = np.squeeze(np.array(out_degree))
        in_degree = np.sum(csr_graph, axis=0, dtype=np.float32)
        in_degree = np.squeeze(np.array(in_degree))

        track_nodes = np.array([0, 1, 2], dtype=np.uint32)
        incrmt_ppr = False

        # extra_param_dict = {}
        # ppr_algo = "power_iteration"
        # ppr_algo = "forward_push"
        # ppr_algo = "ista"
        ppr_algo = "fista"
        
        ista_max_iter: int = 1000
        fista_max_iter: int = 1000
        pwr_iter_max_iter = 1000
        
        # ista_rho: np.float32 = 1e-6
        # ista_early_exit_tol: np.float32 = 1e-7
        
        extra_param_dict = {
            # "ista_rho": ista_rho,
            "power_iteration_max_iter": pwr_iter_max_iter,
            "ista_max_iter": ista_max_iter,
            "fista_max_iter":fista_max_iter,
            # "ista_erly_brk_tol": ista_early_exit_tol,
            "init_epsilon": 1e-10,
        }

        alpha = np.float32(0.2)
        ppr_estimator = DynPPREstimator(
            max_node_num,
            track_nodes,
            alpha,
            ppr_algo,
            incrmt_ppr,
        )
        print(csr_graph.todense())
        print("ppr:", ppr_estimator)

        ppr_estimator.update_ppr(
            csr_graph.indptr,
            csr_graph.indices,
            csr_graph.data,
            out_degree,
            in_degree,
            alpha,
            snapshot_id= 0,
            **extra_param_dict,
        )
        print("ppr:", ppr_estimator)

        edge_events = [
            (0, 2, 1),
            (2, 0, 1),
        ]

        new_row = []
        new_col = []
        new_data = []

        for e_u, e_v, e_delta_w in edge_events:
            # update degree for this edge
            out_degree[e_u] += e_delta_w
            in_degree[e_v] += e_delta_w

            new_row.append(e_u)
            new_col.append(e_v)
            new_data.append(e_delta_w)

            # edge-level adjust
            if ppr_estimator.incrmt_ppr:
                ppr_estimator.dynamic_adjust_ppr_per_edge(
                    e_u,
                    e_v,
                    e_delta_w,
                    out_degree,
                    in_degree,
                    alpha,
                )
            else:
                # DynGraph: non-incremental ppr does not need adjust
                # but re-init ppr_estimator.dict_p_arr/dict_r_arr
                # see: callback_handle_func_all_edge_struct_event_after
                pass

        # graph snapshot-level update
        csr_graph += sp.csr_matrix(
            (new_data, (new_row, new_col)), shape=csr_graph.shape
        )

        if not ppr_estimator.incrmt_ppr:
            ppr_estimator.callback_handle_func_all_edge_struct_event_after(
                None,
                None,
            )

        print(csr_graph.todense())
        print("ppr2:", ppr_estimator)

        # update with ppr_algo
        ppr_estimator.update_ppr(
            csr_graph.indptr,
            csr_graph.indices,
            csr_graph.data,
            out_degree,
            in_degree,
            alpha,
            snapshot_id = 0,
            **extra_param_dict,
        )
        print(csr_graph.todense())
        print("ppr3:", ppr_estimator)


if __name__ == "__main__":
    unittest.main()
