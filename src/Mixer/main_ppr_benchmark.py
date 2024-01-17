import argparse
import itertools
import os
import sys
from os.path import join as os_join
from time import gmtime
from time import process_time as p_time
from time import strftime
import warnings
from typing import Dict
import numpy as np
import numba as nb
import pandas as pd
from pathlib import Path

# from scipy.sparse import csr_matrix

warnings.filterwarnings("ignore")

proj_root_path = Path(__file__).resolve().parents[2]
print(f"proj root path:{proj_root_path}")
# ppv_tmp_dir = "/tmp/"
ppv_tmp_dir = "/mnt/data1/silong/DynMixer-master/tmp/"

sys.path.insert(0, os_join(proj_root_path, "src"))
sys.path.insert(0, os_join(proj_root_path, "src", "DynGL"))
sys.path.insert(0, os_join(proj_root_path, "src", "DynGLReaders"))
from DynGL.DynGraph.DynGraphLoader import DynGraphReader  # noqa: E402
from DynGL.DynPageRank.DynPPRAlgos import PPVResult

from Mixer.PPRGNN_utils import (  # noqa: E402
    setup_seed,
    split_train_dev_test,
    get_dyn_graph_helper,
    get_ppr_estimator,
    set_row_csr,
    get_ppv_cache_hash_dir,
    save_ppv_cache,
    load_ppv_cache,
)


def get_ppr_param_dict(args):
    if args.graph_dataset_name != "ogbn-products":
        if args.graph_dataset_name == "ogbn-arxiv":
            init_epsilon = np.float64(1e-5)
        else:
            init_epsilon = np.float64(1e-7)
        power_iter_gold_epsilon = 1e-3 * init_epsilon
        power_iteration_max_iter = np.uint64(50000)
        ista_max_iter = np.uint64(10000)

        golden_ppr_key = "power-iter-golden"
        ppr_algo_param_dict = {}
        ppr_algo_param_dict["power-iter-golden"] = {
            "ppr_algo": "power_iteration",
            "init_epsilon": power_iter_gold_epsilon,
            "power_iteration_max_iter": power_iteration_max_iter,
            "ista_max_iter": ista_max_iter,
        }

        ppr_algo_param_dict["push"] = {
            "ppr_algo": "forward_push",
            "init_epsilon": init_epsilon,
            "power_iteration_max_iter": power_iteration_max_iter,
            "ista_max_iter": ista_max_iter,
        }

        ppr_algo_param_dict["ista"] = {
            "ppr_algo": "ista",
            "init_epsilon": 0.05 * init_epsilon,
            "power_iteration_max_iter": power_iteration_max_iter,
            "ista_max_iter": ista_max_iter,
        }

    else:
        # for ogbn-products
        # golden_ppr_key = "ista"
        golden_ppr_key = "power-iter-golden"
        ppr_algo_param_dict = {}
        ppr_algo_param_dict["power-iter-golden"] = {
            "ppr_algo": "power_iteration",
            "init_epsilon": np.float64(1e-20),
            "power_iteration_max_iter": np.uint64(50000),
            "ista_max_iter": np.uint64(10000),
        }

        ppr_algo_param_dict["ista"] = {
            "ppr_algo": "ista",
            "init_epsilon": np.float64(1e-10),
            "power_iteration_max_iter": np.uint64(50000),
            "ista_max_iter": np.uint64(5000),
        }

        ppr_algo_param_dict["push"] = {
            "ppr_algo": "forward_push",
            "init_epsilon": np.float64(1e-10),
            "power_iteration_max_iter": np.uint64(50000),
            "ista_max_iter": np.uint64(10000),
        }

    return ppr_algo_param_dict, golden_ppr_key


def main(args):
    # is_verbose = args.use_verbose

    ppr_algo_param_dict, golden_ppr_key = get_ppr_param_dict(args)

    ppv_res_dict: Dict[str, PPVResult] = {}

    local_proj_cache_dir: str = args.local_proj_cache_dir
    exp_name: str = args.exp_name
    for ppr_algo_label, ppr_algo_config in ppr_algo_param_dict.items():
        print("start:", ppr_algo_label)
        ppv_res_dict[ppr_algo_label] = run_ppr_benchmark(
            args,
            ppr_algo_label,
            ppr_algo_config,
        )

        # calculate l1
        num_snapshots = ppv_res_dict[
            ppr_algo_label
        ].update_time_ppr_update.shape[0]

        epprs_path = ppv_res_dict[golden_ppr_key].ppr_arr_snapshot_tmp_file
        apprs_path = ppv_res_dict[ppr_algo_label].ppr_arr_snapshot_tmp_file

        l1_error = cal_snapshot_l1(
            num_snapshots,
            epprs_path,
            apprs_path,
        )
        # if ppr_algo_label != golden_ppr_key:
        #     ppv_res_dict[ppr_algo_label].ppr_arr_per_snapshot = np.array([])
        ppv_res_dict[ppr_algo_label].ppr_l1_err_per_snapshot = l1_error
        print("done:", ppr_algo_label)

    # reset golden ppr
    # ppv_res_dict[golden_ppr_key].ppr_arr_per_snapshot = np.array([])

    # write results/model to out cache folder
    snapshot_res_output_path = os_join(local_proj_cache_dir, exp_name)
    if not os.path.exists(snapshot_res_output_path):
        os.makedirs(snapshot_res_output_path)
    time_suffix = strftime("%Y-%m-%d-%H:%M:%S", gmtime())
    pd_filename = f"df-exp-ppr-benchmark-result-{time_suffix}.pkl"
    result_keys = vars(args)
    result_keys["results"] = ppv_res_dict
    pd_exp_res = pd.DataFrame.from_records([result_keys])
    res_save_path = os_join(snapshot_res_output_path, pd_filename)
    pd_exp_res.to_pickle(res_save_path)
    print(f"result cache file has been saved to {res_save_path}")


def cal_snapshot_l1(
    num_snapshots,
    epprs_path,
    apprs_path,
):
    print("load path:", len(apprs_path.keys()), len(apprs_path.keys()))
    l1_err_per_snapshot = []
    for snapshot_id in range(num_snapshots):
        # load gold_pprs/ apprs
        eppr_ppv_tmp_path = epprs_path[snapshot_id]
        appr_ppv_tmp_path = apprs_path[snapshot_id]
        epprs = load_ppv_cache(None, eppr_ppv_tmp_path, return_dict=True)
        apprs = load_ppv_cache(None, appr_ppv_tmp_path, return_dict=True)
        track_node_ids = list(epprs.keys())
        _l1_err_list = []
        l1_norm_eppr = []
        for node_id in track_node_ids:  # [snapshot_id]
            # calculate the l1 error of one node in every snapshot
            gold_pprs = epprs[node_id]
            appr_pprs = apprs[node_id]
            l1_norm_eppr.append(gold_pprs.sum())
            _l1_err = np.abs(gold_pprs - appr_pprs).sum()
            _l1_err_list.append(_l1_err)
        l1_err_per_snapshot.append(np.mean(_l1_err_list))
        print("eppr l1:", np.mean(l1_norm_eppr))
    # the average l1 error of all tracked nodes in one snapshot
    return np.array(l1_err_per_snapshot, dtype=np.float64)


def run_ppr_benchmark(args, ppr_algo_label, ppr_algo_config):
    args.ppr_algo = ppr_algo_config["ppr_algo"]
    is_verbose = args.use_verbose
    if is_verbose:
        print("args:")
        print(args)
        print("label:", ppr_algo_label)
        print("ppr-param:", ppr_algo_config)

    # dataset related hyper-param
    graph_dataset_name: str = args.graph_dataset_name
    local_proj_data_dir: str = args.local_proj_data_dir
    graph_snapshot_basetime: float = args.graph_snapshot_basetime
    graph_snapshot_interval: float = args.graph_snapshot_interval

    # ppr related hyper-param
    ppr_alpha = args.alpha
    ppr_algo = args.ppr_algo
    is_dangling_avoid = args.is_dangling_avoid
    is_incrmt_ppr = args.use_incrmt_ppr
    # ppr algorithm specific hyper-param
    init_epsilon: np.float64 = ppr_algo_config[
        "init_epsilon"
    ]  # np.float64(1e-10)
    power_iteration_max_iter: np.uint64 = ppr_algo_config[
        "power_iteration_max_iter"
    ]  # 10000
    ista_max_iter: np.uint64 = ppr_algo_config["ista_max_iter"]  # 5000
    # ista_rho: np.float64 = ppr_algo_config["ista_rho"]  # 1e-10
    # early break condition for L1-dis before/after ISTA update (1e-10)
    # ista_early_exit_tol: np.float64 = ppr_algo_config["ista_early_exit_tol"]

    # dataset related hyper-param
    data_strategy = args.data_strategy
    train_per_lb = args.train_per_lb
    dev_per_lb = args.dev_per_lb
    test_per_lb = args.test_per_lb
    total_sampled_node = args.total_sampled_node

    rs = args.rs
    n_cpu = args.n_cpu

    setup_seed(rs)  # deterministic random.
    nb.set_num_threads(n_cpu)

    # init graph snapshots
    dyn_graph, graph_metadata = get_dyn_graph_helper(
        graph_dataset_name=graph_dataset_name,
        local_proj_data_dir=local_proj_data_dir,
        graph_snapshot_basetime=graph_snapshot_basetime,
        graph_snapshot_interval=graph_snapshot_interval,
        is_verbose=is_verbose,
    )

    # init train/dev/test node as empty
    # They are the ppr-tracking nodes, which will be added later.
    train_node_ids = np.array([], dtype=np.uint64)
    dev_node_ids = np.array([], dtype=np.uint64)
    test_node_ids = np.array([], dtype=np.uint64)
    curt_tracked_node_ids = np.hstack(
        (train_node_ids, dev_node_ids, test_node_ids)
    ).astype(np.uint64)

    # init ppr estimator
    ppr_estimator, ppr_extra_param_dict = get_ppr_estimator(
        dyn_graph=dyn_graph,
        alpha=ppr_alpha,
        ppr_algo=ppr_algo,
        incrmt_ppr=is_incrmt_ppr,
        track_nodes=curt_tracked_node_ids,
        init_epsilon=init_epsilon,
        power_iteration_max_iter=power_iteration_max_iter,
        ista_max_iter=ista_max_iter,
    )

    # ppr benchmakr analysis data
    total_snapshots = graph_metadata.total_snapshot
    total_nodes = graph_metadata.total_num_nodes
    ppr_arr_snapshot_tmp_file: Dict[int, str] = {}
    ppr_nnz_per_snapshot: Dict[int, np.ndarray] = {}
    ppr_l1_norm_per_snapshot: Dict[int, np.ndarray] = {}
    ppr_l1_err_per_snapshot: Dict[int, np.ndarray] = {}

    update_time_graph_update = np.zeros(total_snapshots, dtype=float)
    update_time_ppr_update = np.zeros(total_snapshots, dtype=float)
    update_ops_ppr_update = []

    for snapshot_id, (
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
        print(
            "========================="
            f"Snapshot: {snapshot_id}"
            "========================="
        )

        # load a batch of events from cached file.
        # For experiment purpose, we pre-cache the node/edge events
        edge_struct_e = DynGraphReader.load_json_file(edge_struct_path)
        edge_feat_e = DynGraphReader.load_pkl_file(edge_feat_path)
        node_feat_e = DynGraphReader.load_pkl_file(node_feat_path)
        node_lb_e = DynGraphReader.load_pkl_file(node_lb_path)

        # 1: Update graph: node/edge events
        start_t = p_time()
        dyn_graph.update_graph(
            update_timestamp=snapshot_id,
            edge_struct_change=edge_struct_e,
            edge_features_override=edge_feat_e,
            node_feature_override=node_feat_e,
            node_label_override=node_lb_e,
            callback_handle_func_single_edge_struct_event_after=ppr_estimator.callback_handle_func_single_edge_struct_event_after,  # noqa: E501
            callback_handle_func_single_edge_struct_event_before=None,
            callback_handle_func_all_edge_struct_event_after=ppr_estimator.callback_handle_func_all_edge_struct_event_after,  # noqa: E501
        )
        update_time_graph_update[snapshot_id] = p_time() - start_t

        # 1.1: decide which tracked nodes should be train/dev/test
        train_node_ids, dev_node_ids, test_node_ids = split_train_dev_test(
            np.array(list(dyn_graph.node_id_raw_map.keys())),
            dyn_graph.node_labels,
            dyn_graph.degree_in,
            dyn_graph.degree_out,
            train_node_ids,
            dev_node_ids,
            test_node_ids,
            snapshot_id,
            rs,
            data_strategy,
            train_per_lb,
            dev_per_lb,
            test_per_lb,  # can be changed per snapshot.
            total_sampled_node,
            avoid_dangling=is_dangling_avoid,
        )
        curt_tracked_node_ids = np.hstack(
            (train_node_ids, dev_node_ids, test_node_ids)
        ).astype(np.uint64)
        print(
            f"#train: {train_node_ids.shape[0]} "
            f"#dev: {dev_node_ids.shape[0]} "
            f"#test: {test_node_ids.shape[0]} "
            f"#total: {curt_tracked_node_ids.shape[0]} "
        )

        # 1.2: incrementally add tracking nodes.
        ppr_estimator.add_nodes_to_ppr_track(curt_tracked_node_ids)
        # print("only play with 10")
        # ppr_estimator.add_nodes_to_ppr_track(curt_tracked_node_ids[:10])

        # init at first snapshot after all tracked nodes are selected.
        if snapshot_id == 0:
            for node_id in curt_tracked_node_ids:
                # sparse ppv
                # ppr_arr_per_snapshot[node_id] = csr_matrix(
                #     (total_snapshots, total_nodes), dtype=np.float64
                # )
                # ppv nnz
                ppr_nnz_per_snapshot[node_id] = np.zeros(
                    total_snapshots,
                    dtype=int,
                )
                # ppv l1
                ppr_l1_norm_per_snapshot[node_id] = np.zeros(
                    total_snapshots,
                    dtype=np.float64,
                )
                # ppv err compare to golden pwr iter
                ppr_l1_err_per_snapshot[node_id] = np.zeros(
                    total_snapshots,
                    dtype=np.float64,
                )

        # 2: Update PPR to reflect the current node status
        # cache p and data with key in
        # tracked_node_ids, dataset, rs,
        ppv_cache_path = get_ppv_cache_hash_dir(
            args,
            ppv_tmp_dir,
            snapshot_id=snapshot_id,
        )
        ppr_arr_snapshot_tmp_file[snapshot_id] = ppv_cache_path

        start_t = p_time()
        ppr_updates_metric = ppr_estimator.update_ppr(
            dyn_graph.csr_graph.indptr,
            dyn_graph.csr_graph.indices,
            dyn_graph.csr_graph.data,
            dyn_graph.degree_in,
            dyn_graph.degree_out,
            ppr_estimator.alpha,
            snapshot_id,
            **ppr_extra_param_dict,
        )
        update_time_ppr_update[snapshot_id] = p_time() - start_t
        update_ops_ppr_update.append(ppr_updates_metric)

        # inspect_id = 1882
        # inspect_ppv = ppr_estimator.dict_p_arr[inspect_id]
        # print(f"ppv: {inspect_id}", inspect_ppv)
        # print("ppv.sum()", inspect_ppv.sum())

        if os.path.exists(ppv_cache_path):
            print(f"ppv file already exists. Override : {ppv_cache_path}")
        save_ppv_cache(ppr_estimator, ppv_cache_path)

        # store ppv related
        dict_p_arr_at_t = ppr_estimator.dict_p_arr
        for node_id in dict_p_arr_at_t.keys():
            _ppv_dense = np.copy(dict_p_arr_at_t[node_id])
            ppr_nnz_per_snapshot[node_id][snapshot_id] = np.count_nonzero(
                _ppv_dense
            )
            ppr_l1_norm_per_snapshot[node_id][snapshot_id] = np.sum(_ppv_dense)
            # assign ppv row-wise to sparse mat
            # set_row_csr(ppr_arr_per_snapshot[node_id], snapshot_id, _ppv_dense)

    # Gather all results for this snapshot_id:
    return PPVResult(
        ppr_estimator.ppr_algo,
        ppr_estimator.alpha,
        ppr_algo_config,
        # Note: flush it before write to results to save disk.
        # ppr_arr_per_snapshot,
        ppr_arr_snapshot_tmp_file,
        ppr_nnz_per_snapshot,
        ppr_l1_norm_per_snapshot,
        ppr_l1_err_per_snapshot,
        update_time_graph_update,
        update_time_ppr_update,
        update_ops_ppr_update,
    )

    # write results/model to out cache folder
    # snapshot_res_output_path = os_join(local_proj_cache_dir, exp_name)
    # if not os.path.exists(snapshot_res_output_path):
    #     os.makedirs(snapshot_res_output_path)
    # time_suffix = strftime("%Y-%m-%d-%H:%M:%S", gmtime())
    # pd_filename = (
    #     f"df-exp-result-ppr-benchmark-rs-{rs}-{ppr_algo}-{time_suffix}.pkl"
    # )
    # result_keys = vars(args)
    # result_keys["snapshot_results"] = snapshot_stats_record_list
    # pd_exp_res = pd.DataFrame.from_records([result_keys])
    # res_save_path = os_join(snapshot_res_output_path, pd_filename)
    # pd_exp_res.to_pickle(res_save_path)
    # print(f"result cache file has been saved to {res_save_path}")


def args_checker(args):
    pass_args_check: bool = True

    if (
        args.train_per_lb >= 1.0
        and args.train_per_lb >= 1.0
        and args.train_per_lb >= 1.0
        and args.total_sampled_node == 0.0
    ):
        pass_args_check = pass_args_check and True
    elif (
        args.train_per_lb < 1.0
        and args.train_per_lb < 1.0
        and args.train_per_lb < 1.0
        and args.total_sampled_node != 0.0
    ):
        pass_args_check = pass_args_check and True
    else:
        pass_args_check = pass_args_check and False

    return pass_args_check


if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("--exp_name", type=str, default="test")
    arg.add_argument("--graph_dataset_name", type=str, default="cora")
    arg.add_argument(
        "--local_proj_data_dir",
        type=str,
        default="/mnt/data1/silong/DynMixer-master",
    )
    arg.add_argument(
        "--local_proj_cache_dir",
        type=str,
        default="/mnt/data1/silong/DynMixer-master",
    )
    arg.add_argument("--graph_snapshot_basetime", type=float, default=5000.0)
    arg.add_argument("--graph_snapshot_interval", type=float, default=1000.0)

    arg.add_argument("--alpha", type=float, default=0.15)
    arg.add_argument("--is_dangling_avoid", action="store_true")
    arg.add_argument("--use_incrmt_ppr", action="store_true")

    arg.add_argument("--data_strategy", type=str, default="fix-all")
    arg.add_argument("--train_per_lb", type=float, default=0.7)
    arg.add_argument("--dev_per_lb", type=float, default=0.1)
    arg.add_argument("--test_per_lb", type=float, default=0.2)
    arg.add_argument("--total_sampled_node", type=float, default=100.0)

    arg.add_argument("--rs", type=int, default=621)
    arg.add_argument("--use_verbose", action="store_true")
    arg.add_argument("--n_cpu", type=int, default=4)

    args = arg.parse_args()
    assert args_checker(args), "args check does not pass."

    main(args)
