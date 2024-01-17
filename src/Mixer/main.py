import argparse
import itertools
import os
import sys
from os.path import join as os_join
from time import gmtime
from time import process_time as p_time
from time import strftime
from typing import Callable, List
import warnings

import numpy as np
import numba as nb
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

warnings.filterwarnings("ignore")

proj_root_path = Path(__file__).resolve().parents[2]
print(f"proj root path:{proj_root_path}")
# ppv_tmp_dir = "/tmp/"
ppv_tmp_dir = "/mnt/data1/silong/DynMixer-master/tmp/"


sys.path.insert(0, os_join(proj_root_path, "src"))
sys.path.insert(0, os_join(proj_root_path, "src", "DynGL"))
sys.path.insert(0, os_join(proj_root_path, "src", "DynGLReaders"))

from DynGL.DynGraph.DynGraph import DynGraph  # noqa: E402
from DynGL.DynGraph.DynGraphLoader import DynGraphReader  # noqa: E402
from DynGL.DynPageRank import DynPPRAlgos  # noqa: E402
from Mixer.PPRGNN import (  # noqa: E402
    VertexFormer,
    test_model,
    get_model_helper,
    prepare_model_input,
)
from Mixer.PPRGNN_utils import (  # noqa: E402
    ModelPerfResult,
    ModelTrainTimeResult,
    SnapshotTrainEvalRecord,
    EarlyStopper,
    setup_seed,
    split_train_dev_test,
    get_ppv_cache_hash_dir,
    load_ppv_cache,
    save_ppv_cache,
    get_dyn_graph_helper,
    get_ppr_estimator,
)


def train_epochs(
    snapshot_id: int,
    total_snapshots: int,
    model: VertexFormer,
    model_max_train_epochs: int,
    min_epoch_train: int,
    train_node_ids: np.ndarray,
    dev_node_ids: np.ndarray,
    ppr_estimator: DynPPRAlgos.DynPPREstimator,
    dyn_graph: DynGraph,
    ppe_hashcache_id: np.ndarray,
    ppe_hash_cache_sign: np.ndarray,
    ppe_out_dim: int,
    is_cuda_used: bool,
    is_torch_sparse: bool,
    device: str,
    loss_func: Callable,
    optimizer: torch.optim.Optimizer,
    early_stopper: EarlyStopper,
    use_simulated_noise: bool,
    **kwargs,
):
    vertexformer_aggregate_type = model.aggregate_type
    full_feat_mat = dyn_graph.node_feature  # node-id indexed
    full_lb_mat = dyn_graph.node_labels  # node-id indexed

    start_t = p_time()
    former_X_train, target_Y_train, full_feat_mat = prepare_model_input(
        snapshot_id,
        total_snapshots,
        model,
        train_node_ids,
        ppr_estimator,
        ppe_hashcache_id,
        ppe_hash_cache_sign,
        ppe_out_dim,
        is_cuda_used,
        is_torch_sparse,
        device,
        full_feat_mat,
        full_lb_mat,
        use_simulated_noise,
        **kwargs,
    )
    p_time_prepare_input_train = p_time() - start_t

    start_t = p_time()
    former_X_dev, target_Y_dev, full_feat_mat = prepare_model_input(
        snapshot_id,
        total_snapshots,
        model,
        dev_node_ids,
        ppr_estimator,
        ppe_hashcache_id,
        ppe_hash_cache_sign,
        ppe_out_dim,
        is_cuda_used,
        is_torch_sparse,
        device,
        full_feat_mat,
        full_lb_mat,
        use_simulated_noise,
        **kwargs,
    )
    if model.aggregate_type in ("gcn"):
        assert "csr_graph" in kwargs, f"csr_graph not in kwargs"
        csr_graph = kwargs["csr_graph"]
        edge_index = np.array(csr_graph.tocoo().nonzero())
        edge_index = torch.from_numpy(edge_index).to(device, dtype=torch.int)
        # replace nan with 0.0 in full_feat_mat
        if isinstance(full_feat_mat, np.ndarray):
            full_feat_mat = np.nan_to_num(
                full_feat_mat,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            full_feat_mat_tensor = torch.from_numpy(full_feat_mat).to(
                device=device, dtype=torch.float
            )
        else:
            full_feat_mat_tensor = torch.nan_to_num(
                full_feat_mat,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).to(device=device, dtype=torch.float)

    p_time_prepare_input_dev = p_time() - start_t
    p_time_model_per_epoch: List[float] = []
    perf_report_dev_per_epoch: List[ModelPerfResult] = []
    perf_report_train_per_epoch: List[ModelPerfResult] = []

    for curt_epoch_id in range(model_max_train_epochs):
        # eval on both dev/train at begining.
        if model.aggregate_type in ("gcn"):
            perf_report_dev = test_model(
                model,
                former_X_dev,  # not used in gcn
                target_Y_dev,
                loss_func,
                full_feat_mat=full_feat_mat_tensor,
                edge_index=edge_index,
                selected_node_ids=dev_node_ids,
            )
            perf_report_train = test_model(
                model,
                former_X_train,  # not used in gcn
                target_Y_train,
                loss_func,
                full_feat_mat=full_feat_mat_tensor,
                edge_index=edge_index,
                selected_node_ids=train_node_ids,
            )

        else:
            perf_report_dev = test_model(
                model,
                former_X_dev,
                target_Y_dev,
                loss_func,
            )
            perf_report_train = test_model(
                model,
                former_X_train,
                target_Y_train,
                loss_func,
            )

        perf_report_dev_per_epoch.append(perf_report_dev)
        perf_report_train_per_epoch.append(perf_report_train)

        # early stop func
        if early_stopper.should_early_stop(
            curt_epoch_id,
            perf_report_train_per_epoch,
            perf_report_dev_per_epoch,
            min_epoch=min_epoch_train,
        ):
            break

        model.train()
        # record training time
        if is_cuda_used:
            start_t = torch.cuda.Event(enable_timing=True)
            end_t = torch.cuda.Event(enable_timing=True)
            start_t.record()
        else:
            start_t = p_time()

        # train loop
        optimizer.zero_grad()

        if model.aggregate_type in ("gcn"):
            pred_y_logits_train = model(
                full_feat_mat_tensor,
                train_node_ids,
                edge_index,
            )
        else:
            pred_y_logits_train = model(former_X_train)
        ce_loss = loss_func(pred_y_logits_train, target_Y_train)
        ce_loss.backward()
        optimizer.step()
        # if model.aggregate_type == "polyrank":
        #     # print(f"polyrank coefs in training: {model.poly_coefs}")
        #     pass

        print(
            f"training epoch: {curt_epoch_id}/{model_max_train_epochs}",
            end="\r",
        )

        if is_cuda_used:
            end_t.record()
            torch.cuda.synchronize()
            # milliseconds -> seconds
            p_time_model_forbackward = start_t.elapsed_time(end_t)
            p_time_model_forbackward /= 1000.0
        else:
            p_time_model_forbackward = p_time() - start_t
        p_time_model_per_epoch.append(p_time_model_forbackward)

    if curt_epoch_id + 1 == model_max_train_epochs:
        print(
            f"warning: VertexFormer ({vertexformer_aggregate_type}) does  "
            "not converge with model_max_train_epochs: "
            f"{model_max_train_epochs}. "
            "Please consider increase model_max_train_epochs"
        )

    return (
        perf_report_train_per_epoch,
        perf_report_dev_per_epoch,
        p_time_prepare_input_train,
        p_time_prepare_input_dev,
        p_time_model_per_epoch,
    )


def _optimizer_reset(optimizer: torch.optim.Optimizer):
    print("reset optimizer")
    for group in optimizer.param_groups:
        group.update(optimizer.defaults)


def main(args):
    is_verbose = args.use_verbose
    if is_verbose:
        print("args:")
        print(args)

    # dataset related hyper-param
    exp_name: str = args.exp_name
    graph_dataset_name: str = args.graph_dataset_name
    local_proj_data_dir: str = args.local_proj_data_dir
    local_proj_cache_dir: str = args.local_proj_cache_dir
    graph_snapshot_basetime: float = args.graph_snapshot_basetime
    graph_snapshot_interval: float = args.graph_snapshot_interval
    # ppr related hyper-param
    ppr_alpha = args.alpha
    ppr_algo = args.ppr_algo
    pprgo_topk = args.pprgo_topk
    is_dangling_avoid = args.is_dangling_avoid
    ppe_out_dim = args.ppe_out_dim
    is_incrmt_ppr = args.use_incrmt_ppr
    # model related hyper-param
    aggregate_type = args.aggregate_type
    num_mlps = args.num_mlps
    hidden_size_mlps = args.hidden_size_mlps
    drop_r = args.drop_r
    mixrank_mode = args.mixrank_mode
    mixrank_num_samples = args.mixrank_num_samples
    polyrank_mode = args.polyrank_mode
    polyrank_order = args.polyrank_order

    # train related hyper-param
    model_max_train_epochs = args.model_max_train_epochs
    min_epoch_train = args.min_epoch_train
    data_strategy = args.data_strategy
    train_per_lb = args.train_per_lb
    dev_per_lb = args.dev_per_lb
    test_per_lb = args.test_per_lb
    total_sampled_node = args.total_sampled_node
    optim_lr = args.optim_lr
    is_retrain_each_snapshot = args.is_retrain_each_snapshot

    use_simulated_noise = args.use_simulated_noise

    cuda_id = int(args.cuda_id)

    rs = args.rs
    is_cuda_used = args.use_cuda
    is_torch_sparse = args.use_torch_sparse
    is_graph_ppv_saved_in_res = args.use_graph_ppv_debug
    use_ppv_cache = True
    n_cpu = args.n_cpu

    if use_ppv_cache:
        print(
            "Hard coded: use_ppv_cache={use_ppv_cache}. "
            "Methods would use PPV cache to facilitate experiemnts."
        )

    setup_seed(rs)  # deterministic random.
    nb.set_num_threads(n_cpu)  #

    device = "cpu"
    if is_cuda_used:
        assert (
            torch.cuda.is_available()
        ), f"torch.cuda.is_available() = {torch.cuda.is_available()}"
        device = torch.device(f"cuda:{cuda_id}")

    # init graph snapshots
    dyn_graph, graph_metadata = get_dyn_graph_helper(
        graph_dataset_name=graph_dataset_name,
        local_proj_data_dir=local_proj_data_dir,
        graph_snapshot_basetime=graph_snapshot_basetime,
        graph_snapshot_interval=graph_snapshot_interval,
        is_verbose=is_verbose,
    )

    total_snapshots = graph_metadata.total_snapshot

    # init train/dev/test node as empty
    # They are the ppr-tracking nodes, which will be added later.
    train_node_ids = np.array([], dtype=np.uint32)
    dev_node_ids = np.array([], dtype=np.uint32)
    test_node_ids = np.array([], dtype=np.uint32)
    curt_tracked_node_ids = np.hstack(
        (train_node_ids, dev_node_ids, test_node_ids)
    ).astype(np.uint32)

    # init ppr estimator
    ppr_estimator, ppr_extra_param_dict = get_ppr_estimator(
        dyn_graph=dyn_graph,
        alpha=ppr_alpha,
        ppr_algo=ppr_algo,
        incrmt_ppr=is_incrmt_ppr,
        track_nodes=curt_tracked_node_ids,
    )

    # init classification model
    model, ppe_node_id_2_dim_id, ppe_node_id_2_sign = get_model_helper(
        total_num_nodes=graph_metadata.total_num_nodes,
        node_raw_feat_dim=graph_metadata.node_feat_dim,
        ppe_out_dim=ppe_out_dim,
        num_class=graph_metadata.node_label_num_class,
        num_mlps=num_mlps,
        hidden_size_mlps=hidden_size_mlps,
        aggregate_type=aggregate_type,
        rs=rs,
        is_cuda_used=is_cuda_used,
        device=device,
        drop_r=drop_r,
        mixrank_mode=mixrank_mode,
        mixrank_num_samples=mixrank_num_samples,
        polyrank_mode=polyrank_mode,
        polyrank_order=polyrank_order,
    )
    cross_entropy_loss_func = nn.CrossEntropyLoss(
        weight=None,
        size_average=None,
        ignore_index=-100,
        reduction="mean",  # mean none
        # label_smoothing=0.0,
    )

    # Adam SGD will be added in each snapshot.
    assert (
        is_retrain_each_snapshot
    ), "must re-train in each snapshot.set is_retrain_each_snapshot= Ture"

    # train/dev/test output list
    snapshot_stats_record_list = []

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
            "=============================================="
            f"Snapshot: {snapshot_id}/{total_snapshots}"
            "=============================================="
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
        p_time_update_graph = p_time() - start_t

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
        ).astype(np.uint32)
        print(
            f"#train: {train_node_ids.shape[0]} "
            f"#dev: {dev_node_ids.shape[0]} "
            f"#test: {test_node_ids.shape[0]} "
            f"#total: {curt_tracked_node_ids.shape[0]} "
        )

        # 1.2: incrementally add tracking nodes.
        ppr_estimator.add_nodes_to_ppr_track(curt_tracked_node_ids)

        # 2: Update PPR to reflect the current node status
        # cache p and data with key in
        # tracked_node_ids, dataset, rs,

        ppv_cache_path = get_ppv_cache_hash_dir(
            args,
            ppv_tmp_dir,
            snapshot_id=snapshot_id,
        )

        if not os.path.exists(ppv_cache_path) or not use_ppv_cache:
            start_t = p_time()
            ppr_estimator.update_ppr(
                dyn_graph.csr_graph.indptr,
                dyn_graph.csr_graph.indices,
                dyn_graph.csr_graph.data,
                dyn_graph.degree_in,
                dyn_graph.degree_out,
                ppr_estimator.alpha,
                snapshot_id,
                **ppr_extra_param_dict,
            )
            p_time_update_ppr = p_time() - start_t
            # store p and v
            save_ppv_cache(ppr_estimator, ppv_cache_path)
        else:
            load_ppv_cache(ppr_estimator, ppv_cache_path)
            p_time_update_ppr = 0.0

        # 3. train the model
        if is_retrain_each_snapshot:
            setup_seed(rs)  # deterministic random. reset rs state
            model.reset_parameters()
            if model.aggregate_type in ("gcn"):
                optimizer = torch.optim.Adam(
                    model.conv2.parameters(),
                    # [
                    #     {"params": model.conv1.parameters()},
                    #     {"params": model.conv2.parameters()},
                    # ],
                    lr=optim_lr,
                )
            else:
                optimizer = torch.optim.Adam(
                    model.mlps.parameters(), lr=optim_lr
                )
            early_stopper = EarlyStopper(patience=2, min_delta=1e-4)
        else:
            raise NotImplementedError(
                "must re-train each snapshot everytime for benchmark purpose. "
                "please set: is_retrain_each_snapshot = True. "
            )

        (
            perf_report_train_per_epoch,
            perf_report_dev_per_epoch,
            p_time_prepare_input_train,
            p_time_prepare_input_dev,
            p_time_model_per_epoch,
        ) = train_epochs(
            snapshot_id,
            total_snapshots,
            model,  # write the model
            model_max_train_epochs=model_max_train_epochs,
            min_epoch_train=min_epoch_train,
            train_node_ids=train_node_ids,
            dev_node_ids=dev_node_ids,
            ppr_estimator=ppr_estimator,
            dyn_graph=dyn_graph,
            ppe_hashcache_id=ppe_node_id_2_dim_id,
            ppe_hash_cache_sign=ppe_node_id_2_sign,
            ppe_out_dim=ppe_out_dim,
            is_cuda_used=is_cuda_used,
            is_torch_sparse=is_torch_sparse,
            device=device,
            loss_func=cross_entropy_loss_func,
            optimizer=optimizer,
            pprgo_topk=pprgo_topk,
            early_stopper=early_stopper,
            use_simulated_noise=use_simulated_noise,
            csr_graph=dyn_graph.csr_graph,
        )

        train_time_result = ModelTrainTimeResult(
            p_time_update_graph,
            p_time_update_ppr,
            p_time_prepare_input_train,
            p_time_prepare_input_dev,
            p_time_model_per_epoch,
        )

        # 5. evaluate on testing set
        former_X_test, target_Y_test, full_feat_mat = prepare_model_input(
            snapshot_id,
            total_snapshots,
            model,
            test_node_ids,
            ppr_estimator,
            ppe_node_id_2_dim_id,
            ppe_node_id_2_sign,
            ppe_out_dim,
            is_cuda_used,
            is_torch_sparse,
            device,
            dyn_graph.node_feature,
            dyn_graph.node_labels,
            pprgo_topk=pprgo_topk,
            use_simulated_noise=use_simulated_noise,
        )

        if model.aggregate_type in ("gcn"):
            edge_index = np.array(dyn_graph.csr_graph.tocoo().nonzero())
            edge_index = torch.from_numpy(edge_index).to(
                device, dtype=torch.int
            )
            # replace nan with 0.0 in full_feat_mat
            if isinstance(full_feat_mat, np.ndarray):
                full_feat_mat = np.nan_to_num(
                    full_feat_mat,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                full_feat_mat_tensor = torch.from_numpy(full_feat_mat).to(
                    device=device, dtype=torch.float
                )
            else:
                full_feat_mat_tensor = torch.nan_to_num(
                    full_feat_mat,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).to(device=device, dtype=torch.float)

            perf_report_test = test_model(
                model,
                former_X_test,  # not used in gcn
                target_Y_test,
                cross_entropy_loss_func,
                full_feat_mat=full_feat_mat_tensor,
                edge_index=edge_index,
                selected_node_ids=test_node_ids,
            )

        else:
            perf_report_test = test_model(
                model,
                former_X_test,
                target_Y_test,
                cross_entropy_loss_func,
            )

        # Gather all results for this snapshot_id:
        graph_ppv_dict = {}
        if is_graph_ppv_saved_in_res:
            print(
                "Saving ppv into SnapshotTrainEvalRecord, "
                "which is used to plot ppv changes over time. "
                "It may cost much larger disk usage."
            )
            for k in ppr_estimator.dict_p_arr.keys():
                graph_ppv_dict[k] = ppr_estimator.dict_p_arr[k].tolist()

        snapshot_stats_record = SnapshotTrainEvalRecord(
            snapshot_id,
            perf_report_train_per_epoch,
            perf_report_dev_per_epoch,
            perf_report_test,
            train_time_result,
            graph_ppv_dict,
        )

        print(
            "train_loss",
            len(perf_report_train_per_epoch),
            " -> ".join(
                [
                    f"{i}: {_.loss} "
                    for i, _ in enumerate(perf_report_train_per_epoch)
                ]
            ),
        )
        snapshot_stats_record_list.append(snapshot_stats_record)

    # write results/model to out cache folder
    snapshot_res_output_path = os_join(local_proj_cache_dir, exp_name)
    if not os.path.exists(snapshot_res_output_path):
        os.makedirs(snapshot_res_output_path)
    model_setting = (
        f"{aggregate_type}-{ppr_algo}-{mixrank_mode}-{polyrank_mode}"
    )
    time_suffix = strftime("%Y-%m-%d-%H:%M:%S", gmtime())
    pd_filename = f"df-exp-result-{model_setting}-{time_suffix}.pkl"
    result_keys = vars(args)
    result_keys["snapshot_results"] = snapshot_stats_record_list
    pd_exp_res = pd.DataFrame.from_records([result_keys])
    res_save_path = os_join(snapshot_res_output_path, pd_filename)
    pd_exp_res.to_pickle(res_save_path)
    print(f"result cache file has been saved to {res_save_path}")


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

    # check mixrank config
    if args.aggregate_type == "mixrank":
        if args.mixrank_mode == "null":
            pass_args_check = pass_args_check and False

        if args.mixrank_num_samples > 0:
            pass_args_check = pass_args_check and True
        else:
            pass_args_check = pass_args_check and False

    # check polyrank config
    if args.aggregate_type == "polyrank_mode":
        if args.polyrank_mode == "null":
            pass_args_check = pass_args_check and False
        if args.polyrank_order > 0:
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
    arg.add_argument("--aggregate_type", type=str, default="mlp")
    arg.add_argument("--pprgo_topk", type=int, default=16)
    arg.add_argument("--alpha", type=float, default=0.15)
    arg.add_argument("--ppr_algo", type=str, default="ista")
    arg.add_argument("--is_dangling_avoid", action="store_true")

    arg.add_argument("--ppe_out_dim", type=int, default=128)
    arg.add_argument("--use_incrmt_ppr", action="store_true")

    arg.add_argument("--num_mlps", type=int, default=2)
    arg.add_argument(
        "--hidden_size_mlps",
        help="delimited input with ,",
        type=lambda s: [int(item) for item in s.split(",")],
        default="32,16",
    )
    arg.add_argument("--drop_r", type=float, default=0.15)
    arg.add_argument("--mixrank_mode", type=str, default="null")
    arg.add_argument("--mixrank_num_samples", type=int, default=0)
    arg.add_argument("--polyrank_mode", type=str, default="null")
    arg.add_argument("--polyrank_order", type=int, default=0)

    arg.add_argument("--optim_lr", type=float, default=0.05)

    arg.add_argument("--model_max_train_epochs", type=int, default=100)
    arg.add_argument("--min_epoch_train", type=int, default=5)

    arg.add_argument("--data_strategy", type=str, default="fix-all")
    arg.add_argument("--train_per_lb", type=float, default=10.0)
    arg.add_argument("--dev_per_lb", type=float, default=20.0)
    arg.add_argument("--test_per_lb", type=float, default=20.0)
    arg.add_argument("--total_sampled_node", type=float, default=0.0)
    arg.add_argument("--is_retrain_each_snapshot", action="store_true")

    arg.add_argument("--rs", type=int, default=621)
    arg.add_argument("--use_cuda", action="store_true")
    arg.add_argument("--use_torch_sparse", action="store_true")
    arg.add_argument("--use_verbose", action="store_true")
    arg.add_argument("--use_graph_ppv_debug", action="store_true")

    arg.add_argument("--use_simulated_noise", action="store_true")

    arg.add_argument("--n_cpu", type=int, default=4)
    arg.add_argument("--cuda_id", type=int, default=0)

    args = arg.parse_args()
    assert args_checker(args), "args check does not pass."
    main(args)
    # see scripts/run-exp-1.sh
