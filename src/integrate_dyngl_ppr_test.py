import itertools
import unittest
import numpy as np
import numba as nb
import sys
from time import perf_counter as p_time

from typing import Dict, List

sys.path.insert(0, "/mnt/data1/silong/DynMixer-master/src/DynGL")
sys.path.insert(0, "/mnt/data1/silong/DynMixer-master/src/DynGLReaders")

from DynGL.DynGraph.DynGraph import DynGraph
from DynGL.DynGraph.DynGraphLoader import DynGraphMetadata, DynGraphReader
from DynGL.DynPageRank import DynPPRAlgos
from DynGL.DynPageRank.DynPPRAlgos import PPVResult
from DynGLReaders.DynGraphReaders import (
    DynGraphReaderKarate,
    DynGraphReaderPlanetoid,
)


nb.set_num_threads(1)  #


class test_integrate_dyngraph_ppr(unittest.TestCase):
    def test_on_cora(self):
        print("============ cora ============")

        local_data_karate_path = "/mnt/data1/silong/DynMixer-master/"

        cora_dataset_reader = DynGraphReaderPlanetoid(
            graph_dataset_name="cora",
            local_dataset_dir_abs_path=local_data_karate_path,
            verbose=False,
        )
        cora_dataset_reader.download_parse_sort_data()
        graph_metadata: DynGraphMetadata = (
            cora_dataset_reader.get_graph_event_snapshots_from_sorted_events(
                interval=1000.0,
                base_snapshot_t=5000.0,
            )
        )

        # track_nodes = np.array(
        #     [0, 1, 2, 3, 4, 5],
        #     dtype=np.uint32,
        # )
        track_nodes = np.arange(
            4,
            dtype=np.uint32,
        )
        alpha = np.float32(0.2)
        init_epsilon = np.float32(1e-8)
        incrmt_ppr = True

        final_ppr_res = {}

        ppr_algo = "power_iteration"
        extra_param_dict = {
            "init_epsilon": np.float32(1e-20),  # set to be very small
            "power_iteration_max_iter": np.uint32(10000),
        }

        final_ppr_res[ppr_algo] = self.run_helper(
            graph_metadata,
            track_nodes,
            ppr_algo,
            extra_param_dict,
            alpha,
            return_ppv_result=True,
            incrmt_ppr=True,
        )

        ppr_algo = "ista"
        ista_max_iter: int = 5000
        ista_rho: np.float32 = 1e-12
        ista_early_exit_tol: np.float32 = 1e-20  # l1
        extra_param_dict = {
            "init_epsilon": init_epsilon,
            "ista_rho": ista_rho,
            "ista_max_iter": ista_max_iter,
            "ista_erly_brk_tol": ista_early_exit_tol,
        }
        final_ppr_res[ppr_algo] = self.run_helper(
            graph_metadata,
            track_nodes,
            ppr_algo,
            extra_param_dict,
            alpha,
            return_ppv_result=True,
            incrmt_ppr=incrmt_ppr,
        )

        ppr_algo = "forward_push"
        extra_param_dict = {
            "init_epsilon": init_epsilon,
        }
        final_ppr_res[ppr_algo] = self.run_helper(
            graph_metadata,
            track_nodes,
            ppr_algo,
            extra_param_dict,
            alpha,
            return_ppv_result=True,
            incrmt_ppr=incrmt_ppr,
        )

        for algo in final_ppr_res.keys():
            self.__cal_ppr_l1_err_inplace(final_ppr_res, algo, track_nodes)
            print(f"=========={algo}==========")
            # print("-- ppr nnz:")
            # print(final_ppr_res[algo].ppr_nnz_per_snapshot)
            print("-- ppr l1 error:")
            print(final_ppr_res[algo].ppr_l1_err_per_snapshot)
            print("-- edge update time(sec)|include p/r adjust for dyn push):")
            print(final_ppr_res[algo].update_time_graph_update)
            print("-- appr update time(sec):")
            print(final_ppr_res[algo].update_time_ppr_update)

    def skip_test_on_citseer(self):
        print("============ citeseer ============")

        local_data_karate_path = "/mnt/data1/silong/DynMixer-master/"

        graph_dataset_reader = DynGraphReaderPlanetoid(
            graph_dataset_name="citeseer",
            local_dataset_dir_abs_path=local_data_karate_path,
            verbose=False,
        )
        graph_dataset_reader.download_parse_sort_data()
        graph_metadata: DynGraphMetadata = (
            graph_dataset_reader.get_graph_event_snapshots_from_sorted_events(
                interval=100.0,
                base_snapshot_t=8000.0,
            )
        )

        track_nodes = np.arange(
            20,
            dtype=np.uint32,
        )
        alpha = np.float32(0.5)
        init_epsilon = np.float32(1e-10)
        incrmt_ppr = False

        final_ppr_res = {}

        ppr_algo = "power_iteration"
        extra_param_dict = {
            "init_epsilon": np.float32(1e-20),  # set to be very small
            "power_iteration_max_iter": np.uint32(10000),
        }
        final_ppr_res[ppr_algo] = self.run_helper(
            graph_metadata,
            track_nodes,
            ppr_algo,
            extra_param_dict,
            alpha,
            return_ppv_result=True,
            incrmt_ppr=incrmt_ppr,
        )

        ppr_algo = "ista"
        ista_max_iter: int = 5000
        ista_rho: np.float32 = 1e-10
        ista_early_exit_tol: np.float32 = 1e-20  # l1
        extra_param_dict = {
            "init_epsilon": init_epsilon,
            "ista_rho": ista_rho,
            "ista_max_iter": ista_max_iter,
            "ista_erly_brk_tol": ista_early_exit_tol,
        }
        final_ppr_res[ppr_algo] = self.run_helper(
            graph_metadata,
            track_nodes,
            ppr_algo,
            extra_param_dict,
            alpha,
            return_ppv_result=True,
            incrmt_ppr=incrmt_ppr,
        )

        ppr_algo = "forward_push"
        extra_param_dict = {
            "init_epsilon": init_epsilon,
        }
        final_ppr_res[ppr_algo] = self.run_helper(
            graph_metadata,
            track_nodes,
            ppr_algo,
            extra_param_dict,
            alpha,
            return_ppv_result=True,
            incrmt_ppr=incrmt_ppr,
        )

        for algo in final_ppr_res.keys():
            self.__cal_ppr_l1_err_inplace(final_ppr_res, algo, track_nodes)
            print(f"=========={algo}==========")
            # print("-- ppr nnz:")
            # print(final_ppr_res[algo].ppr_nnz_per_snapshot)
            print("-- ppr l1 error:")
            print(final_ppr_res[algo].ppr_l1_err_per_snapshot)
            print("-- edge update time(sec)|include p/r adjust for dyn push):")
            print(final_ppr_res[algo].update_time_graph_update)
            print("-- appr update time(sec):")
            print(final_ppr_res[algo].update_time_ppr_update)

    def __cal_ppr_l1_err_inplace(
        self,
        final_ppr_res,
        algo,
        track_nodes,
        gold_algo="power_iteration",
    ):
        assert (
            gold_algo in final_ppr_res.keys()
        ), f"{gold_algo} does not exist. We use {gold_algo} as gold ppr"

        l1_err_node_list = []
        for node_id in track_nodes:
            # calculate the l1 error of one node in every snapshot
            gold_pprs = final_ppr_res[gold_algo].ppr_arr_per_snapshot[node_id]
            appr_pprs = final_ppr_res[algo].ppr_arr_per_snapshot[node_id]
            l1_err = np.squeeze(
                np.array(np.abs(gold_pprs - appr_pprs).sum(axis=1))
            )
            l1_err_node_list.append(l1_err)
        l1_err_node_list = np.array(l1_err_node_list)
        final_ppr_res[algo].ppr_l1_err_per_snapshot = np.mean(
            l1_err_node_list, axis=0
        )  # the average l1 error of all tracked nodes in one snapshot

    def skip_test_on_cora_dyn_track(self):
        print(
            "============ cora (dynamically add new tracking nodes)"
            " ============"
        )

        local_data_karate_path = "/mnt/data1/silong/DynMixer-master/"

        cora_dataset_reader = DynGraphReaderPlanetoid(
            graph_dataset_name="cora",
            local_dataset_dir_abs_path=local_data_karate_path,
            verbose=False,
        )
        cora_dataset_reader.download_parse_sort_data()
        graph_metadata: DynGraphMetadata = (
            cora_dataset_reader.get_graph_event_snapshots_from_sorted_events(
                interval=100.0,
                base_snapshot_t=5000.0,
            )
        )

        track_nodes_per_snapshot: List[np.ndarray] = []
        for _i in range(graph_metadata.total_snapshot):
            temp_ids = np.arange((_i + 1) * 1, dtype=np.uint32)
            track_nodes_per_snapshot.append(temp_ids)
            # print(f"tracking {temp_ids.shape[0]} nodes: {temp_ids}")

        alpha = np.float32(0.2)
        init_epsilon = np.float32(1e-8)
        incrmt_ppr = True

        final_ppr_res = {}

        ppr_algo = "power_iteration"
        extra_param_dict = {
            "init_epsilon": np.float32(1e-20),  # set to be very small
            "power_iteration_max_iter": np.uint32(10000),
        }

        final_ppr_res[ppr_algo] = self.run_helper_dyn_track(
            graph_metadata,
            track_nodes_per_snapshot,
            ppr_algo,
            extra_param_dict,
            alpha,
            return_ppv_result=True,
            incrmt_ppr=incrmt_ppr,
        )

        ppr_algo = "ista"
        ista_max_iter: int = 5000
        ista_rho: np.float32 = 1e-12
        ista_early_exit_tol: np.float32 = 1e-20  # l1
        extra_param_dict = {
            "init_epsilon": init_epsilon,
            "ista_rho": ista_rho,
            "ista_max_iter": ista_max_iter,
            "ista_erly_brk_tol": ista_early_exit_tol,
        }
        final_ppr_res[ppr_algo] = self.run_helper_dyn_track(
            graph_metadata,
            track_nodes_per_snapshot,
            ppr_algo,
            extra_param_dict,
            alpha,
            return_ppv_result=True,
            incrmt_ppr=incrmt_ppr,
        )

        ppr_algo = "forward_push"
        extra_param_dict = {
            "init_epsilon": init_epsilon,
        }
        final_ppr_res[ppr_algo] = self.run_helper_dyn_track(
            graph_metadata,
            track_nodes_per_snapshot,
            ppr_algo,
            extra_param_dict,
            alpha,
            return_ppv_result=True,
            incrmt_ppr=incrmt_ppr,
        )

        for algo in final_ppr_res.keys():
            self.__cal_ppr_l1_err_inplace_dyn_track(
                final_ppr_res, algo, track_nodes_per_snapshot
            )
            print(f"=========={algo}==========")
            # print("-- ppr nnz:")
            # print(final_ppr_res[algo].ppr_nnz_per_snapshot)
            print("-- ppr l1 error:")
            print(final_ppr_res[algo].ppr_l1_err_per_snapshot)
            print("-- edge update time(sec)|include p/r adjust for dyn push):")
            print(final_ppr_res[algo].update_time_graph_update)
            print("-- appr update time(sec):")
            print(final_ppr_res[algo].update_time_ppr_update)

    def __cal_ppr_l1_err_inplace_dyn_track(
        self,
        final_ppr_res,
        algo,
        track_nodes_per_snapshot,
        gold_algo="power_iteration",
    ):
        assert (
            gold_algo in final_ppr_res.keys()
        ), f"{gold_algo} does not exist. We use {gold_algo} as gold ppr"

        l1_err_per_snapshot = []

        num_snapshots = len(track_nodes_per_snapshot)

        for snapshot_id in range(num_snapshots):
            _l1_err_list = []
            for node_id in track_nodes_per_snapshot[snapshot_id]:
                # calculate the l1 error of one node in every snapshot
                gold_pprs = final_ppr_res[gold_algo].ppr_arr_per_snapshot[
                    node_id
                ][snapshot_id]

                appr_pprs = final_ppr_res[algo].ppr_arr_per_snapshot[node_id][
                    snapshot_id
                ]
                _l1_err = np.abs(gold_pprs - appr_pprs).sum()
                _l1_err_list.append(_l1_err)

            l1_err_per_snapshot.append(np.mean(_l1_err_list))

        final_ppr_res[algo].ppr_l1_err_per_snapshot = np.array(
            l1_err_per_snapshot
        )  # the average l1 error of all tracked nodes in one snapshot

    def skip_test_on_karate(self):
        local_data_karate_path = "/mnt/data1/silong/DynMixer-master/"
        karate_dataset_reader = DynGraphReaderKarate(
            graph_dataset_name="karate",
            local_dataset_dir_abs_path=local_data_karate_path,
            verbose=False,
        )
        karate_dataset_reader.download_parse_sort_data()
        graph_metadata: DynGraphMetadata = (
            karate_dataset_reader.get_graph_event_snapshots_from_sorted_events(
                interval=4.0,
                base_snapshot_t=50.0,
            )
        )

        track_nodes = np.array(
            [
                0,
            ],
            dtype=np.uint32,
        )
        alpha = np.float32(0.2)
        incrmt_ppr = True

        final_ppr_res = {}

        ppr_algo = "power_iteration"
        extra_param_dict = {}
        final_ppr_res[ppr_algo] = self.run_helper(
            graph_metadata,
            track_nodes,
            ppr_algo,
            extra_param_dict,
            alpha,
            return_ppv_result=True,
            incrmt_ppr=incrmt_ppr,
        )

        ppr_algo = "ista"
        ista_max_iter: int = 100
        ista_rho: np.float32 = 1e-6
        extra_param_dict = {
            "ista_rho": ista_rho,
            "ista_max_iter": ista_max_iter,
        }
        final_ppr_res[ppr_algo] = self.run_helper(
            graph_metadata,
            track_nodes,
            ppr_algo,
            extra_param_dict,
            alpha,
            return_ppv_result=True,
            incrmt_ppr=incrmt_ppr,
        )

        ppr_algo = "forward_push"
        extra_param_dict = {}
        final_ppr_res[ppr_algo] = self.run_helper(
            graph_metadata,
            track_nodes,
            ppr_algo,
            extra_param_dict,
            alpha,
            return_ppv_result=True,
            incrmt_ppr=incrmt_ppr,
        )

        for algo in final_ppr_res.keys():
            print(final_ppr_res[algo].update_time_graph_update)

    def run_helper(
        self,
        graph_metadata: DynGraphMetadata,
        track_nodes: np.ndarray,
        ppr_algo: str,
        extra_param_dict: Dict,
        alpha: float,
        return_ppv_result: bool = False,
        incrmt_ppr: bool = True,
    ):
        dyn_graph = DynGraph(
            max_node_num=graph_metadata.total_num_nodes,
            node_feat_dim=graph_metadata.node_feat_dim,
            edge_feat_dim=2,
            node_label_num_class=graph_metadata.node_label_num_class,
        )

        ppr_estimator = DynPPRAlgos.DynPPREstimator(
            dyn_graph.max_node_num,
            track_nodes,
            alpha,
            ppr_algo,
            incrmt_ppr,
        )

        # analysis data
        total_snapshots = graph_metadata.total_snapshot
        total_nodes = graph_metadata.total_num_nodes
        ppr_arr_per_snapshot: Dict[int, np.ndarray] = {}
        ppr_nnz_per_snapshot: Dict[int, np.ndarray] = {}
        ppr_l1_norm_per_snapshot: Dict[int, np.ndarray] = {}
        ppr_l1_err_per_snapshot: Dict[int, np.ndarray] = {}
        for node_id in track_nodes:
            ppr_arr_per_snapshot[node_id] = np.zeros(
                (total_snapshots, total_nodes),
                dtype=np.float32,
            )
            ppr_nnz_per_snapshot[node_id] = np.zeros(
                total_snapshots,
                dtype=int,
            )
            ppr_l1_norm_per_snapshot[node_id] = np.zeros(
                total_snapshots,
                dtype=float,
            )
            ppr_l1_err_per_snapshot[node_id] = np.zeros(
                total_snapshots,
                dtype=float,
            )
        update_time_graph_update = np.zeros(total_snapshots, dtype=float)
        update_time_ppr_update = np.zeros(total_snapshots, dtype=float)

        # into the graph events loop
        for timestamp, (
            edge_struct_path,
            edge_feat_path,
            node_feat_path,
            node_lb_path,
        ) in enumerate(
            itertools.zip_longest(
                graph_metadata.edge_event_split_path_list,
                [],
                graph_metadata.node_feature_event_split_path_list,
                graph_metadata.node_label_snap_file_path_list,
                fillvalue=None,
            )
        ):
            # a batch of events
            edge_struct_e = DynGraphReader.load_json_file(edge_struct_path)
            edge_feat_e = DynGraphReader.load_pkl_file(edge_feat_path)
            node_feat_e = DynGraphReader.load_pkl_file(node_feat_path)
            node_lb_e = DynGraphReader.load_pkl_file(node_lb_path)

            # apply the batch of events, and fire callback functions at
            # before/after batch updates
            # before/after per edge struct changes.
            # the callback function for updating edge-level happens here.
            # csr + csr happens in update_graph()
            start_t = p_time()
            dyn_graph.update_graph(
                update_timestamp=timestamp,
                edge_struct_change=edge_struct_e,
                edge_features_override=edge_feat_e,
                node_feature_override=node_feat_e,
                node_label_override=node_lb_e,
                callback_handle_func_single_edge_struct_event_after=ppr_estimator.callback_handle_func_single_edge_struct_event_after,
                callback_handle_func_single_edge_struct_event_before=None,
                callback_handle_func_all_edge_struct_event_after=ppr_estimator.callback_handle_func_all_edge_struct_event_after,
            )
            update_time_graph_update[timestamp] = p_time() - start_t

            # Update PPR
            start_t = p_time()
            ppr_estimator.update_ppr(
                dyn_graph.csr_graph.indptr,
                dyn_graph.csr_graph.indices,
                dyn_graph.csr_graph.data,
                dyn_graph.degree_in,
                dyn_graph.degree_out,
                alpha,
                **extra_param_dict,
            )
            update_time_ppr_update[timestamp] = p_time() - start_t

            # analysis
            dict_p_arr_at_t = ppr_estimator.dict_p_arr
            for tracked_node_id in dict_p_arr_at_t.keys():
                # print(
                #     "ppv nnz",
                #     tracked_node_id,
                #     np.count_nonzero(dict_p_arr_at_t[tracked_node_id]),
                # )
                _ppv = dict_p_arr_at_t[tracked_node_id]
                ppr_arr_per_snapshot[tracked_node_id][timestamp] = np.copy(
                    _ppv
                )
                ppr_nnz_per_snapshot[tracked_node_id][
                    timestamp
                ] = np.count_nonzero(_ppv)
                ppr_l1_norm_per_snapshot[tracked_node_id][timestamp] = np.sum(
                    _ppv
                )
                ppr_l1_err_per_snapshot[tracked_node_id][timestamp] = -99.0

        # print(dict_p_arr_at_t)

        # print(ppr_estimator.ppr_algo)
        # print(ppr_estimator.alpha)
        # print(ppr_estimator.dict_p_arr)
        if return_ppv_result:
            return PPVResult(
                ppr_estimator.ppr_algo,
                ppr_estimator.alpha,
                extra_param_dict,
                ppr_arr_per_snapshot,
                ppr_nnz_per_snapshot,
                ppr_l1_norm_per_snapshot,
                ppr_l1_err_per_snapshot,
                update_time_graph_update,
                update_time_ppr_update,
            )

    def run_helper_dyn_track(
        self,
        graph_metadata: DynGraphMetadata,
        track_nodes_per_snapshot: List[np.ndarray],
        ppr_algo: str,
        extra_param_dict: Dict,
        alpha: float,
        return_ppv_result: bool = False,
        incrmt_ppr: bool = True,
    ):
        dyn_graph = DynGraph(
            max_node_num=graph_metadata.total_num_nodes,
            node_feat_dim=graph_metadata.node_feat_dim,
            edge_feat_dim=2,
            node_label_num_class=graph_metadata.node_label_num_class,
        )

        tracked_nodes_first_snapshot = track_nodes_per_snapshot[0]
        tracked_nodes_all_snapshot = np.hstack(
            track_nodes_per_snapshot
        ).astype(np.uint32)
        tracked_nodes_all_snapshot = np.unique(tracked_nodes_all_snapshot)
        ppr_estimator = DynPPRAlgos.DynPPREstimator(
            dyn_graph.max_node_num,
            tracked_nodes_first_snapshot,
            alpha,
            ppr_algo,
            incrmt_ppr,
        )

        # analysis data
        total_snapshots = graph_metadata.total_snapshot
        total_nodes = graph_metadata.total_num_nodes
        ppr_arr_per_snapshot: Dict[int, np.ndarray] = {}
        ppr_nnz_per_snapshot: Dict[int, np.ndarray] = {}
        ppr_l1_norm_per_snapshot: Dict[int, np.ndarray] = {}
        ppr_l1_err_per_snapshot: Dict[int, np.ndarray] = {}
        for node_id in tracked_nodes_all_snapshot:
            ppr_arr_per_snapshot[node_id] = np.zeros(
                (total_snapshots, total_nodes),
                dtype=np.float32,
            )
            ppr_nnz_per_snapshot[node_id] = np.zeros(
                total_snapshots,
                dtype=int,
            )
            ppr_l1_norm_per_snapshot[node_id] = np.zeros(
                total_snapshots,
                dtype=float,
            )
            ppr_l1_err_per_snapshot[node_id] = np.zeros(
                total_snapshots,
                dtype=float,
            )
        update_time_graph_update = np.zeros(total_snapshots, dtype=float)
        update_time_ppr_update = np.zeros(total_snapshots, dtype=float)

        # into the graph events loop
        for timestamp, (
            edge_struct_path,
            edge_feat_path,
            node_feat_path,
            node_lb_path,
        ) in enumerate(
            itertools.zip_longest(
                graph_metadata.edge_event_split_path_list,
                [],
                graph_metadata.node_feature_event_split_path_list,
                graph_metadata.node_label_snap_file_path_list,
                fillvalue=None,
            )
        ):
            # a batch of events
            edge_struct_e = DynGraphReader.load_json_file(edge_struct_path)
            edge_feat_e = DynGraphReader.load_pkl_file(edge_feat_path)
            node_feat_e = DynGraphReader.load_pkl_file(node_feat_path)
            node_lb_e = DynGraphReader.load_pkl_file(node_lb_path)

            # apply the batch of events, and fire callback functions at
            # before/after batch updates
            # before/after per edge struct changes.
            # the callback function for updating edge-level happens here.
            # csr + csr happens in update_graph()
            start_t = p_time()
            dyn_graph.update_graph(
                update_timestamp=timestamp,
                edge_struct_change=edge_struct_e,
                edge_features_override=edge_feat_e,
                node_feature_override=node_feat_e,
                node_label_override=node_lb_e,
                callback_handle_func_single_edge_struct_event_after=ppr_estimator.callback_handle_func_single_edge_struct_event_after,
                callback_handle_func_single_edge_struct_event_before=None,
                callback_handle_func_all_edge_struct_event_after=ppr_estimator.callback_handle_func_all_edge_struct_event_after,
            )
            update_time_graph_update[timestamp] = p_time() - start_t

            newly_tracked_node_ids = track_nodes_per_snapshot[timestamp]
            ppr_estimator.add_nodes_to_ppr_track(newly_tracked_node_ids)

            print(
                f"#Tracked-nodes: {ppr_estimator.track_nodes.shape[0]} "
                f"at snapshot {timestamp}"
            )

            # Update PPR
            start_t = p_time()
            ppr_estimator.update_ppr(
                dyn_graph.csr_graph.indptr,
                dyn_graph.csr_graph.indices,
                dyn_graph.csr_graph.data,
                dyn_graph.degree_in,
                dyn_graph.degree_out,
                alpha,
                **extra_param_dict,
            )
            update_time_ppr_update[timestamp] = p_time() - start_t

            # analysis
            dict_p_arr_at_t = ppr_estimator.dict_p_arr
            for tracked_node_id in dict_p_arr_at_t.keys():
                # print(
                #     "ppv nnz",
                #     tracked_node_id,
                #     np.count_nonzero(dict_p_arr_at_t[tracked_node_id]),
                # )
                _ppv = dict_p_arr_at_t[tracked_node_id]
                ppr_arr_per_snapshot[tracked_node_id][timestamp] = np.copy(
                    _ppv
                )
                ppr_nnz_per_snapshot[tracked_node_id][
                    timestamp
                ] = np.count_nonzero(_ppv)
                ppr_l1_norm_per_snapshot[tracked_node_id][timestamp] = np.sum(
                    _ppv
                )
                ppr_l1_err_per_snapshot[tracked_node_id][timestamp] = -99.0

        # print(dict_p_arr_at_t)

        # print(ppr_estimator.ppr_algo)
        # print(ppr_estimator.alpha)
        # print(ppr_estimator.dict_p_arr)
        if return_ppv_result:
            return PPVResult(
                ppr_estimator.ppr_algo,
                ppr_estimator.alpha,
                extra_param_dict,
                ppr_arr_per_snapshot,
                ppr_nnz_per_snapshot,
                ppr_l1_norm_per_snapshot,
                ppr_l1_err_per_snapshot,
                update_time_graph_update,
                update_time_ppr_update,
            )


if __name__ == "__main__":
    unittest.main()
