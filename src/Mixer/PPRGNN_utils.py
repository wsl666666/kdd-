import numpy as np
from sklearn.utils import murmurhash3_32 as murmurhash
import numba
import torch
import random
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import Dict, List, Tuple
import hashlib
from os.path import join as os_join
import json
import pickle
import os
import sys
from pathlib import Path
import scipy.sparse as sp

proj_root_path = Path(__file__).resolve().parents[2]
print(f"proj root path:{proj_root_path}")

sys.path.insert(0, os_join(proj_root_path, "src"))
sys.path.insert(0, os_join(proj_root_path, "src", "DynGL"))
sys.path.insert(0, os_join(proj_root_path, "src", "DynGLReaders"))

from DynGL.DynGraph.DynGraph import DynGraph  # noqa: E402
from DynGL.DynPageRank import DynPPRAlgos  # noqa: E402
from DynGL.DynGraph.DynGraphLoader import (  # noqa: E402
    DynGraphMetadata,
)
from DynGLReaders.DynGraphReaders import (  # noqa: E402
    DynGraphReaderPlanetoid,
    DynGraphReaderFlicker,
    DynGraphReaderReddit2,
    DynGraphReaderOGB,
    DynGraphReaderEllipticBitcoin,
    DynGraphReaderAttributedGraph,
    DynGraphReaderWikiCS,
    DynGraphReaderCitationFull,
    DynGraphReaderCoauthor,
    DynGraphReaderHeterophilyActor,
    DynGraphReaderHeterophilyWiki,
)


@dataclass_json
@dataclass
class ModelPerfResult:
    loss: float
    roc_auc: float
    clf_report: Dict


@dataclass_json
@dataclass
class ModelTrainTimeResult:
    graph_update_time: float
    ppr_update_time: float
    model_prepare_input_train: float
    model_prepare_input_dev: float
    model_forback_per_epoch: List[float]


@dataclass_json
@dataclass
class SnapshotTrainEvalRecord:
    snapshot_id: int
    perf_report_train_per_epoch: List[ModelPerfResult]
    perf_report_dev_per_epoch: List[ModelPerfResult]
    perf_report_test: ModelPerfResult
    train_time_result: ModelTrainTimeResult
    graph_ppv_dict: Dict


def get_hash_LUT(n: int, dim: int = 512, rnd_seed: int = 0):
    """get the cache of dim-sign, dim-map from hash function

    Args:
        n (int): the total number of input dim size (ppr vector length)
        dim (int, optional): the out dimension. Defaults to 512.
        rnd_seed (int, optional): the hash random seed. Defaults to 0.

    Returns:
        np.ndarray: the dimension mapping of ppr-vector to out-vector
        np.ndarray: the sign (+/-) mapping of ppr-vector to out-vector
    """

    node_id_2_dim_id: np.ndarray = np.zeros(n, dtype=np.int32)
    node_id_2_sign: np.ndarray = np.zeros(n, dtype=np.int8)
    for _ in range(n):
        dim_id = murmurhash(_, seed=rnd_seed, positive=True) % dim
        sign = murmurhash(_, seed=rnd_seed, positive=True) % 2
        node_id_2_dim_id[_] = dim_id
        node_id_2_sign[_] = 1 if sign == 1 else -1
    return node_id_2_dim_id, node_id_2_sign


@numba.njit(cache=True, parallel=True, fastmath=True, nogil=True)
def get_hash_embed(
    node_id_2_dim_id: np.ndarray,
    node_id_2_sign: np.ndarray,
    out_dim: int,
    q_nodes: np.ndarray,
    ppv_dim: int,
    indices: np.ndarray,
    indptr: np.ndarray,
    data: np.ndarray,
):
    """get hash embeddings from ppvs and pre-cached dimension/sign
    mapping.
    The input indices/indptr/data is from the row-sliced csr_mat.

    Args:
        node_id_2_dim_id (np.ndarray): the dimension mapping of ppr
            vector to out-vector
        node_id_2_sign (np.ndarray): the sign (+/-) mapping of ppr
            vector to out-vector
        out_dim (int): the dimension of output embedding vectors.
        q_nodes (np.ndarray): the queried node ids.
        ppv_dim (int): the dimension of original ppr vector
        indices (np.ndarray): the csr.indices of ppr matrix
        indptr (np.ndarray): the csr.indptr of ppr matrix
        data (np.ndarray): the csr.data of ppr matrix

    Returns:
        np.ndarray: the queried hash embeddings from ppr.
    """
    emb_mat: np.ndarray = np.zeros(
        (q_nodes.shape[0], out_dim),
        dtype=np.float64,
    )
    for i in numba.prange(q_nodes.shape[0]):  # for all nodes.
        js = indices[indptr[i] : indptr[i + 1]]
        vals = data[indptr[i] : indptr[i + 1]]
        # emb_vec = emb_mat[i, :]
        for j, val in zip(js, vals):
            _map_out_dim = node_id_2_dim_id[j]
            # emb_vec[_map_out_dim]
            emb_mat[i, _map_out_dim] += node_id_2_sign[j] * np.maximum(
                np.float64(0.0),
                np.float64(np.log(val * ppv_dim)),
            )
    return emb_mat


def setup_seed(seed: int):
    """For benchmark/reproducibility purpose: set all seed and make
    torch.cudnn determinstic

    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def split_train_dev_test(
    graph_curt_all_nodes_ids: np.ndarray,
    graph_curt_all_nodes_lbs: np.ndarray,
    degree_in: np.ndarray,
    degree_out: np.ndarray,
    train_node_ids: np.ndarray,
    dev_node_ids: np.ndarray,
    test_node_ids: np.ndarray,
    snapshot_id: int,
    rs: int,
    data_strategy: str = "fix-all",
    train_per_lb: int = 5,
    dev_per_lb: int = 5,
    test_per_lb: int = 5,
    total_sampled_node: int = 500,
    avoid_dangling: bool = True,
):
    """get train/dev/test for each snapshot
    Ensure:
        1. three sets are disjoint.
        2. train set has label-balanced samples wrt train_per_lb.
        3. dev/test has random sample after selecting train samples.
        4. the request train/dev/test sets have specified #samples.

    """

    rnd_state = np.random.default_rng(seed=rs)

    num_node_lb_class = graph_curt_all_nodes_lbs.shape[1]
    num_lbs_per_node = np.count_nonzero(graph_curt_all_nodes_lbs, axis=1)
    num_node_with_lbs = np.count_nonzero(num_lbs_per_node)
    num_graph_nodes = graph_curt_all_nodes_ids.shape[0]

    if num_node_with_lbs != num_graph_nodes:
        print(
            f"num_node_of_lbs ({num_node_with_lbs}) != num_graph_nodes"
            f" ({num_graph_nodes}). not every node has label?"
        )
    # assert num_node_with_lbs == num_graph_nodes, (
    #     f"num_node_of_lbs ({num_node_with_lbs}) != num_graph_nodes"
    #     f" ({num_graph_nodes}). not every node has label?"
    # )

    # sample train nodes for those having labels
    lb_nnz_nodes = np.nonzero(num_lbs_per_node)[0]
    lb_nnz_y_mat = graph_curt_all_nodes_lbs[lb_nnz_nodes, :]
    lb_nnz_y = np.nonzero(lb_nnz_y_mat)[1]
    # first row: node id of nnz lb
    # secoid row: its lb (assuming it's multi-class classification)
    node_lb_pairs_all = np.vstack((lb_nnz_nodes, lb_nnz_y))

    if avoid_dangling:
        # filter out based on in/out-degree
        lb_nnz_nodes_out_d = degree_out[lb_nnz_nodes]
        non_dangling_mask = lb_nnz_nodes_out_d > 0.0
        node_lb_pairs = node_lb_pairs_all[:, non_dangling_mask]
    else:
        raise NotImplementedError(
            "dangling nodes should be ignored, or process properly"
        )

    # At very first snapshot:
    # split curt_tracked node ids into train/dev/test
    # train_set has label-balanced samples.
    # dev/test are random shuffled and label imbalabnced.
    if snapshot_id == 0:
        if train_per_lb >= 1.0 and dev_per_lb >= 1.0 and test_per_lb >= 1.0:
            # print(
            #     "Info: balanced sampling: "
            #     f"{train_per_lb}/{dev_per_lb}/{test_per_lb} "
            #     "per label for train/dev/test."
            # )
            (
                train_all_node_ids,
                dev_all_node_ids,
                test_all_node_ids,
            ) = get_train_dev_test_by_per_lb(
                train_per_lb,
                dev_per_lb,
                test_per_lb,
                rnd_state,
                num_node_lb_class,
                node_lb_pairs,
            )
            __check_datasets_size(
                train_node_ids,
                dev_node_ids,
                test_node_ids,
                train_per_lb,
                dev_per_lb,
                test_per_lb,
                num_node_lb_class,
                graph_curt_all_nodes_lbs,
            )

        else:
            print(
                "Info: stratified sampling "
                f"{train_per_lb}/{dev_per_lb}/{test_per_lb} "
                "*100% per for train/dev/test. "
            )
            (
                train_all_node_ids,
                dev_all_node_ids,
                test_all_node_ids,
            ) = get_train_dev_test_by_stratify(
                train_per_lb,
                dev_per_lb,
                test_per_lb,
                rnd_state,
                num_node_lb_class,
                node_lb_pairs,
                total_sampled_node,
            )

        train_node_ids = np.hstack(train_all_node_ids)
        dev_node_ids = np.hstack(dev_all_node_ids)
        test_node_ids = np.hstack(test_all_node_ids)

    if data_strategy == "fix-all":
        # do nothing.
        pass

    elif data_strategy == "add-test":
        # random sample extra testing and add to testing set.
        # get disjoint nodes
        curt_num_test_node_ids = test_node_ids.shape[0]
        num_new_samples = test_per_lb - curt_num_test_node_ids
        assert num_new_samples > 0, (
            f"curt_test_node: {curt_num_test_node_ids} while "
            f"add-test requests total: {test_per_lb}."
            "the request test samples should be strictly, monotonicaly"
            "increasing!"
        )

        curt_tracked_node_ids = np.hstack(
            (train_node_ids, dev_node_ids, test_node_ids)
        ).astype(np.uint32)
        new_candidate = np.setdiff1d(
            node_lb_pairs[0, :],
            curt_tracked_node_ids,
        )
        new_test_add = rnd_state.permutation(new_candidate)[:num_new_samples]
        test_node_ids = np.hstack((test_node_ids, new_test_add))
    else:
        raise NotImplementedError

    return train_node_ids, dev_node_ids, test_node_ids


def get_train_dev_test_by_per_lb(
    train_per_lb,
    dev_per_lb,
    test_per_lb,
    rnd_state,
    num_node_lb_class,
    node_lb_pairs,
):
    """get node ids for train/dev/test set per label as balance dataset"""
    train_all_node_ids = []
    dev_all_node_ids = []
    test_all_node_ids = []

    for node_lb in range(num_node_lb_class):
        idx_match_lb = np.where(node_lb_pairs[1, :] == node_lb)[0]
        node_id_match_lb = node_lb_pairs[0, :][idx_match_lb]
        # shuffle node ids of this label
        node_id_match_lb = rnd_state.permutation(node_id_match_lb)

        # split for training nodes
        __sample_nodes_per_lb(
            0,
            train_per_lb,
            train_all_node_ids,
            node_lb,
            node_id_match_lb,
            "train",
        )
        # split for dev nodes
        __sample_nodes_per_lb(
            train_per_lb,
            train_per_lb + dev_per_lb,
            dev_all_node_ids,
            node_lb,
            node_id_match_lb,
            "dev",
        )
        # split for test nodes
        __sample_nodes_per_lb(
            train_per_lb + dev_per_lb,
            train_per_lb + dev_per_lb + test_per_lb,
            test_all_node_ids,
            node_lb,
            node_id_match_lb,
            "test",
        )

    return train_all_node_ids, dev_all_node_ids, test_all_node_ids


def __sample_nodes_per_lb(
    start_idx,
    end_idx,
    all_node_ids,
    node_lb,
    node_id_match_lb,
    set_name,
    is_strict: bool = True,
):
    if is_strict:
        assert node_id_match_lb.shape[0] >= end_idx, (
            f"{set_name} label "
            f"{node_lb} (shape[0] = {node_id_match_lb.shape[0]}) cannot "
            f"fullfill the sliced range start:{start_idx},  end:{end_idx}"
        )
    elif node_id_match_lb.shape[0] < end_idx:
        print(
            f"Warning the fetched data size is smaller for {set_name} label "
            f"{node_lb} (shape[0] = {node_id_match_lb.shape[0]}) cannot "
            f"fullfill the sliced range start:{start_idx},  end:{end_idx}"
        )

    all_node_ids.append(node_id_match_lb[start_idx:end_idx])


def get_train_dev_test_by_stratify(
    train_per_lb,
    dev_per_lb,
    test_per_lb,
    rnd_state,
    num_node_lb_class,
    node_lb_pairs,
    total_sampled_node: int = 500,
):
    """get node ids for train/dev/test as stratified dataset.
    train_per_lb/dev_per_lb/test_per_lb < 1.0

    """
    train_all_node_ids = []
    dev_all_node_ids = []
    test_all_node_ids = []

    total_nodes_with_labels = node_lb_pairs.shape[1]

    for node_lb in range(num_node_lb_class):
        idx_match_lb = np.where(node_lb_pairs[1, :] == node_lb)[0]
        node_ids_per_lb = node_lb_pairs[0, :][idx_match_lb]
        # shuffle node ids of this label
        node_ids_per_lb = rnd_state.permutation(node_ids_per_lb)
        total_num_samples_per_lb = node_ids_per_lb.shape[0]
        node_num_sample_per_lb = np.ceil(
            (total_num_samples_per_lb / total_nodes_with_labels)
            * total_sampled_node
        )

        _train_num_per_lb = np.ceil(
            node_num_sample_per_lb * train_per_lb,
        ).astype(int)
        _dev_num_per_lb = np.ceil(
            node_num_sample_per_lb * dev_per_lb,
        ).astype(int)
        _test_num_per_lb = np.ceil(
            node_num_sample_per_lb * test_per_lb,
        ).astype(int)
        _total_num_per_lb = (
            _train_num_per_lb + _dev_num_per_lb + _test_num_per_lb
        )
        print(
            f"lb: {node_lb}, {_train_num_per_lb}, {_dev_num_per_lb},"
            f" {_test_num_per_lb}, {_total_num_per_lb}"
        )

        assert _train_num_per_lb >= 1, f"num train {_train_num_per_lb} <1"
        assert _dev_num_per_lb >= 1, f"num dev {_dev_num_per_lb} <1"
        assert _test_num_per_lb >= 1, f"num test {_test_num_per_lb} <1"

        # TODO: Skip this class if there is not enough nodes.
        # if the class is very imbalanced (e.g., amazon-product graph)
        # where several label has #nodes<3. We sample no node
        if total_num_samples_per_lb >= _total_num_per_lb:
            pass  # okay
        else:
            print(
                f"Label {node_lb} has been excluded from train/dev/test."
                f"total requested samples {_total_num_per_lb} > total data"
                f" {total_num_samples_per_lb}"
            )
        # assert total_num_samples_per_lb >= _total_num_per_lb, (
        #     f"total requested samples {_total_num_per_lb} > total data"
        #     f" {total_num_samples_per_lb}"
        # )

        train_all_node_ids.append(node_ids_per_lb[0:_train_num_per_lb])
        dev_all_node_ids.append(
            node_ids_per_lb[
                _train_num_per_lb : _train_num_per_lb + _dev_num_per_lb
            ]
        )
        test_all_node_ids.append(
            node_ids_per_lb[
                (_train_num_per_lb + _dev_num_per_lb) : (
                    _train_num_per_lb + _dev_num_per_lb + _test_num_per_lb
                )
            ]
        )
    return train_all_node_ids, dev_all_node_ids, test_all_node_ids


def __show_label_hist(node_ids, graph_curt_all_nodes_lbs):
    lb_y = np.nonzero(graph_curt_all_nodes_lbs[node_ids])[1]
    assert node_ids.shape[0] == lb_y.shape[0], (
        f"#node ids: {node_ids.shape[0]} != #label-ids {lb_y.shape[0]}. "
        "Any node has multi-labels?"
    )
    unique, counts = np.unique(lb_y, return_counts=True)
    return np.asarray((unique, counts)).T


def __check_datasets_size(
    train_node_ids,
    dev_node_ids,
    test_node_ids,
    train_per_lb,
    dev_per_lb,
    test_per_lb,
    num_node_lb_class,
    graph_curt_all_nodes_lbs,
):
    assert train_node_ids.shape[0] == num_node_lb_class * train_per_lb, (
        "Total balanced train set"
        f" (#sample:{num_node_lb_class* train_per_lb}) cannot be created"
        f" with actual data #sample: {train_node_ids.shape[0]}. "
    )
    assert dev_node_ids.shape[0] <= num_node_lb_class * dev_per_lb, (
        f"Total dev set (#sample:{dev_per_lb} * {num_node_lb_class}) cannot"
        f" be created with actual data #sample: {dev_node_ids.shape[0]}. "
    )
    assert test_node_ids.shape[0] <= num_node_lb_class * test_per_lb, (
        f"Total test set (#sample:{test_per_lb} * {num_node_lb_class})"
        " cannot be created with actual data #sample:"
        f" {test_node_ids.shape[0]}. "
    )

    # print(f"train:{train_node_ids}")
    # print(f"dev:{dev_node_ids}")
    # print(f"test:{test_node_ids}")

    # train_lbs_count = __show_label_hist(
    #     train_node_ids,
    #     graph_curt_all_nodes_lbs,
    # )
    # dev_lbs_count = __show_label_hist(
    #     dev_node_ids,
    #     graph_curt_all_nodes_lbs,
    # )
    # test_lbs_count = __show_label_hist(
    #     test_node_ids,
    #     graph_curt_all_nodes_lbs,
    # )

    # print(f"train lbs: {train_lbs_count}")
    # print(f"dev lbs: {dev_lbs_count}")
    # print(f"test lbs: {test_lbs_count}")


def get_ppv_cache_hash_dir(args, tmp_dir, **kwargs):
    # test
    d = vars(args)
    hash_dict = {}
    hash_keys = [
        # "aggregate_type",
        "alpha",
        "data_strategy",
        "graph_dataset_name",
        "graph_snapshot_basetime",
        "graph_snapshot_interval",
        "is_dangling_avoid",
        "ppr_algo",
        "rs",
        "dev_per_lb",
        "test_per_lb",
        "total_sampled_node",
        "train_per_lb",
        "use_incrmt_ppr",
    ]
    for k in hash_keys:
        hash_dict[k] = d[k]
    args_js = json.dumps({**hash_dict, **kwargs}, sort_keys=True, default=str)
    hash_file = hashlib.md5(args_js.encode("utf-8")).hexdigest()
    proj_name = "DynMixer"
    if not os.path.exists(os_join(tmp_dir, proj_name)):
        os.makedirs(os_join(tmp_dir, proj_name))
    cache_file = os_join(tmp_dir, proj_name, hash_file + ".pkl")
    return cache_file


def load_ppv_cache(ppr_estimator, ppv_cache_path, return_dict: bool = False):
    # print(f"Info: load dict_p_arr from {ppv_cache_path}")
    with open(ppv_cache_path, "rb") as f:
        cache_dict = pickle.load(f)
    dict_p_arr_norm = cache_dict["dict_p_arr"]
    dict_p_arr = {}
    if ppr_estimator is None and return_dict:
        # print("ppr_estimator == None, return dict ")
        for k, v in dict_p_arr_norm.items():
            dict_p_arr[np.uint32(k)] = np.array(
                v.todense().flatten(),
                dtype=np.float64,
            )
        return dict_p_arr
    for k, v in dict_p_arr_norm.items():
        ppr_estimator.dict_p_arr[np.uint32(k)] = np.array(
            v.todense().flatten(),
            dtype=np.float64,
        )


def save_ppv_cache(ppr_estimator, ppv_cache_path):
    print(f"Info: dump dict_p/r_arr to {ppv_cache_path}")
    cache_dict = {}
    dict_p_arr_norm = {}
    for k in ppr_estimator.dict_p_arr.keys():
        dict_p_arr_norm[k] = sp.csr_array(ppr_estimator.dict_p_arr[k])
    cache_dict["dict_p_arr"] = dict_p_arr_norm
    with open(ppv_cache_path, "wb") as f:
        pickle.dump(cache_dict, f)


class EarlyStopper:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def _early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        # elif validation_loss > (self.min_validation_loss + self.min_delta):
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def should_early_stop(
        self,
        curt_epoch_id: int,
        perf_report_train_per_epoch: List[ModelPerfResult],
        perf_report_dev_per_epoch: List[ModelPerfResult],
        min_epoch: int = 5,
    ) -> bool:
        """Early stop decision function based on loss on dev set."""
        if curt_epoch_id == 0 or curt_epoch_id < min_epoch:
            # at epoch 0: no actual training happened
            # since we test model first.
            return False

        # prev_dev_res = perf_report_dev_per_epoch[curt_epoch_id - 1]
        curt_dev_res = perf_report_dev_per_epoch[curt_epoch_id]
        return self._early_stop(curt_dev_res.loss)

    def should_early_stop_old(
        curt_epoch_id: int,
        perf_report_train_per_epoch: List[ModelPerfResult],
        perf_report_dev_per_epoch: List[ModelPerfResult],
        min_epoch: int = 5,
    ) -> bool:
        """Early stop decision function based on loss on dev set."""
        if curt_epoch_id == 0 or curt_epoch_id < min_epoch:
            # for very first epoch, keep training
            # at epoch 0: no actual training happened
            # since we test model first.
            return False

        prev_dev_res = perf_report_dev_per_epoch[curt_epoch_id - 1]
        curt_dev_res = perf_report_dev_per_epoch[curt_epoch_id]

        # the simplest design:
        # if dev loss start to increase, early stop.
        prev_dev_loss = prev_dev_res.loss
        curt_dev_loss = curt_dev_res.loss
        if curt_dev_loss >= prev_dev_loss:
            return True
        else:
            return False


def topk_norm(
    max_top_k: int,
    matrix_input: np.ndarray,
    ord_norm: int = 1,
) -> torch.tensor:
    """take top-k element per row and re-normalize row-wise as per the
    ord (e.g., l-1 norm)

    """

    out_mat = np.zeros_like(matrix_input)

    m = out_mat.shape[1]
    if max_top_k > m:
        max_top_k = m
    sort_k = m - max_top_k
    # mask lowest vals per row
    large_idx = np.argpartition(matrix_input, sort_k, axis=1)[:, sort_k:]
    row_idx = np.arange(matrix_input.shape[0]).reshape(-1, 1)
    out_mat[row_idx, large_idx] = matrix_input[row_idx, large_idx]
    l1_norm = np.linalg.norm(out_mat, axis=1, keepdims=True, ord=ord_norm)
    out_norm_mat = (out_mat / l1_norm).astype(np.float64)
    out_norm_mat = torch.tensor(out_norm_mat, dtype=torch.float64)

    return out_norm_mat


def bottomk_norm(
    max_btm_k: int,
    matrix_input: np.ndarray,
    ord_norm: int = 1,
) -> torch.tensor:
    """take bottom-k element per row and re-normalize row-wise as per the
    ord (e.g., l-1 norm)
    ignore zero entries

    """
    out_mat = np.zeros_like(matrix_input)
    matrix_input = np.array(matrix_input, copy=True)
    matrix_input[matrix_input == 0.0] = np.Inf

    m = out_mat.shape[1]
    if max_btm_k > m:
        max_btm_k = m

    sort_k = max_btm_k  # m - btm_k
    # mask lowest vals per row
    small_idx = np.argpartition(matrix_input, sort_k, axis=1)[:, :sort_k]
    row_idx = np.arange(matrix_input.shape[0]).reshape(-1, 1)
    out_mat[row_idx, small_idx] = matrix_input[row_idx, small_idx]
    # in-case of nnz is smaller than requested btm-k
    out_mat[out_mat == np.Inf] = 0.0
    l1_norm = np.linalg.norm(out_mat, axis=1, keepdims=True, ord=ord_norm)
    out_norm_mat = (out_mat / l1_norm).astype(np.float64)
    out_norm_mat = torch.tensor(out_norm_mat, dtype=torch.float64)
    return out_norm_mat


def randomk_norm(
    max_rnd_k: int,
    matrix_input: np.ndarray,
    ord_norm: int = 1,
) -> torch.tensor:
    """take random-k element per row and re-normalize row-wise as per the
    ord (e.g., l-1 norm)
    ignore zero entries

    """
    out_mat = np.zeros_like(matrix_input)
    matrix_input = np.array(matrix_input, copy=True)

    m = out_mat.shape[1]
    if max_rnd_k > m:
        max_rnd_k = m

    # random sample nnzs
    random_nnz_idx = np.zeros((matrix_input.shape[0], max_rnd_k), dtype=int)
    for i in range(matrix_input.shape[0]):
        nnz_idx = np.nonzero(matrix_input[i])[0]
        if nnz_idx.shape[0] == max_rnd_k:
            for j in range(nnz_idx.shape[0]):
                random_nnz_idx[i, j] = nnz_idx[j]
        elif nnz_idx.shape[0] > max_rnd_k:
            random_nnz_idx[i] = np.random.choice(
                nnz_idx, size=max_rnd_k, replace=False
            )
        else:
            random_nnz_idx[i] = np.random.choice(
                nnz_idx, size=max_rnd_k, replace=True
            )

    row_idx = np.arange(matrix_input.shape[0]).reshape(-1, 1)
    out_mat[row_idx, random_nnz_idx] = matrix_input[row_idx, random_nnz_idx]

    l1_norm = np.linalg.norm(out_mat, axis=1, keepdims=True, ord=ord_norm)
    out_norm_mat = (out_mat / l1_norm).astype(np.float64)
    out_norm_mat = torch.tensor(out_norm_mat, dtype=torch.float64)
    return out_norm_mat


def get_dyn_graph_helper(
    graph_dataset_name: str = "cora",
    local_proj_data_dir: str = "/home/xingzguo/projects_data/DynMixer/",
    graph_snapshot_basetime: float = 5000.0,
    graph_snapshot_interval: float = 100.0,
    is_verbose: bool = False,
) -> Tuple[DynGraph, DynGraphMetadata]:
    if graph_dataset_name in ("cora", "citeseer", "pubmed"):
        dataset_reader = DynGraphReaderPlanetoid(
            graph_dataset_name=graph_dataset_name,
            local_dataset_dir_abs_path=local_proj_data_dir,
            verbose=is_verbose,
            use_undirect=True,
        )
    elif graph_dataset_name in ("flickr"):
        dataset_reader = DynGraphReaderFlicker(
            graph_dataset_name=graph_dataset_name,
            local_dataset_dir_abs_path=local_proj_data_dir,
            verbose=is_verbose,
            use_undirect=True,
        )
    elif graph_dataset_name in ("reddit2"):
        dataset_reader = DynGraphReaderReddit2(
            graph_dataset_name=graph_dataset_name,
            local_dataset_dir_abs_path=local_proj_data_dir,
            verbose=is_verbose,
            use_undirect=True,
        )
    elif graph_dataset_name in ("ogbn-arxiv"):
        dataset_reader = DynGraphReaderOGB(
            graph_dataset_name=graph_dataset_name,
            local_dataset_dir_abs_path=local_proj_data_dir,
            verbose=is_verbose,
            use_undirect=True,
        )
    elif graph_dataset_name in ("ogbn-products"):
        dataset_reader = DynGraphReaderOGB(
            graph_dataset_name=graph_dataset_name,
            local_dataset_dir_abs_path=local_proj_data_dir,
            verbose=is_verbose,
            already_sort=True,
            use_undirect=True,
        )
    elif graph_dataset_name in ("bitcoin"):
        dataset_reader = DynGraphReaderEllipticBitcoin(
            graph_dataset_name=graph_dataset_name,
            local_dataset_dir_abs_path=local_proj_data_dir,
            verbose=is_verbose,
            use_undirect=True,
        )
    elif graph_dataset_name in ("BlogCatalog"):
        dataset_reader = DynGraphReaderAttributedGraph(
            graph_dataset_name=graph_dataset_name,
            local_dataset_dir_abs_path=local_proj_data_dir,
            verbose=is_verbose,
            use_undirect=True,
        )
    elif graph_dataset_name in ("WikiCS"):
        dataset_reader = DynGraphReaderWikiCS(
            graph_dataset_name=graph_dataset_name,
            local_dataset_dir_abs_path=local_proj_data_dir,
            verbose=is_verbose,
            use_undirect=True,
        )
    elif graph_dataset_name in ("DBLP"):
        dataset_reader = DynGraphReaderCitationFull(
            graph_dataset_name=graph_dataset_name,
            local_dataset_dir_abs_path=local_proj_data_dir,
            verbose=is_verbose,
            use_undirect=True,
        )
    elif graph_dataset_name in ("Physics", "CS"):
        dataset_reader = DynGraphReaderCoauthor(
            graph_dataset_name=graph_dataset_name,
            local_dataset_dir_abs_path=local_proj_data_dir,
            verbose=is_verbose,
            use_undirect=True,
        )
    # elif graph_dataset_name in ("film"):
    #     dataset_reader = DynGraphReaderHeterophilyActor(
    #         graph_dataset_name=graph_dataset_name,
    #         local_dataset_dir_abs_path=local_proj_data_dir,
    #         verbose=is_verbose,
    #     )
    # elif graph_dataset_name in ("chameleon", "squirrel"):
    #     dataset_reader = DynGraphReaderHeterophilyWiki(
    #         graph_dataset_name=graph_dataset_name,
    #         local_dataset_dir_abs_path=local_proj_data_dir,
    #         verbose=is_verbose,
    #     )
    else:
        raise NotImplementedError(f"{graph_dataset_name} not found.")

    dataset_reader.download_parse_sort_data()
    graph_metadata: DynGraphMetadata = (
        dataset_reader.get_graph_event_snapshots_from_sorted_events(
            interval=graph_snapshot_interval,
            base_snapshot_t=graph_snapshot_basetime,
        )
    )

    dyn_graph = DynGraph(
        max_node_num=graph_metadata.total_num_nodes,
        node_feat_dim=graph_metadata.node_feat_dim,
        edge_feat_dim=2,
        node_label_num_class=graph_metadata.node_label_num_class,
    )
    return dyn_graph, graph_metadata


def get_ppr_estimator(
    dyn_graph: DynGraph,
    alpha: np.float64,
    ppr_algo: str,
    incrmt_ppr: bool,
    track_nodes: np.ndarray,
    init_epsilon: np.float64 = np.float64(1e-5),
    power_iteration_max_iter: np.uint32 = np.uint32(10000),
    ista_max_iter: int = 5000,
) -> Tuple[DynPPRAlgos.DynPPREstimator, Dict]:
    """Return the DynPPREstimator and extra input when updates ppr"""

    ppr_estimator = DynPPRAlgos.DynPPREstimator(
        dyn_graph.max_node_num,
        track_nodes,
        alpha,
        ppr_algo,
        incrmt_ppr,
    )

    ppr_extra_param_dict = {}
    if ppr_algo == "power_iteration":
        ppr_extra_param_dict = {
            "init_epsilon": init_epsilon,
            "power_iteration_max_iter": power_iteration_max_iter,
        }
    elif ppr_algo == "ista":
        ppr_extra_param_dict = {
            "init_epsilon": init_epsilon,
            "ista_max_iter": ista_max_iter,
        }
    elif ppr_algo == "forward_push":
        ppr_extra_param_dict = {
            "init_epsilon": init_epsilon,
        }
    elif ppr_algo == "fista":
        ppr_extra_param_dict = {
            "init_epsilon": init_epsilon,
        }    
    else:
        raise NotImplementedError

    return ppr_estimator, ppr_extra_param_dict


def set_row_csr(A, row_idx, new_row):
    """
    Replace a row in a CSR sparse matrix A.

    Parameters
    ----------
    A: csr_matrix
        Matrix to change
    row_idx: int
        index of the row to be changed
    new_row: np.array
        list of new values for the row of A

    Returns
    -------
    None (the matrix A is changed in place)

    Prerequisites
    -------------
    The row index shall be smaller than the number of rows in A
    The number of elements in new row must be equal to the number of columns in matrix A
    # SEE: https://stackoverflow.com/questions/28427236/set-row-of-csr-matrix

    """
    assert sp.isspmatrix_csr(A), "A shall be a csr_matrix"
    assert row_idx < A.shape[0], (
        "The row index ({0}) shall be smaller than the number of rows in A"
        " ({1})".format(row_idx, A.shape[0])
    )
    try:
        N_elements_new_row = len(new_row)
    except TypeError:
        msg = (
            "Argument new_row shall be a list or numpy array, is now a {0}"
            .format(type(new_row))
        )
        raise AssertionError(msg)
    N_cols = A.shape[1]
    assert N_cols == N_elements_new_row, (
        "The number of elements in new row ({0}) must be equal to "
        "the number of columns in matrix A ({1})".format(
            N_elements_new_row, N_cols
        )
    )

    idx_start_row = A.indptr[row_idx]
    idx_end_row = A.indptr[row_idx + 1]
    additional_nnz = N_cols - (idx_end_row - idx_start_row)

    A.data = np.r_[A.data[:idx_start_row], new_row, A.data[idx_end_row:]]
    A.indices = np.r_[
        A.indices[:idx_start_row], np.arange(N_cols), A.indices[idx_end_row:]
    ]
    A.indptr = np.r_[
        A.indptr[: row_idx + 1], A.indptr[(row_idx + 1) :] + additional_nnz
    ]


def simulate_additive_noise(
    original_feat_mat: np.ndarray,
    snapshot_id,
    total_snapshot,
    # base_original_sign: float = 0.2, # defualt
    # noise_mean_scale: float = 5.0,  # default it 5
    # noise_std_scale: float = 5.0,
    base_original_sign: float = 0.4,  # for arvix
    noise_mean_scale: float = 1.0,  # default it 2 for arxiv
    noise_std_scale: float = 1.0,
):
    # Simulate
    # number of features
    original_feat = np.copy(original_feat_mat)
    num_row, num_col = original_feat.shape
    origin_p = base_original_sign + (1 - base_original_sign) * (
        snapshot_id
    ) / (total_snapshot - 1)
    noise_p = 1.0 - origin_p
    mu = np.mean(original_feat, axis=1)
    nnz_ids = ~np.isnan(mu)  # nan if node has not arrived yet.
    # print(nnz_ids)
    # print("see", original_feat[nnz_ids])
    mu_nnz = np.mean(original_feat[nnz_ids])
    std_nnz = np.std(original_feat[nnz_ids])
    print(mu_nnz, std_nnz)
    # make noise stronger
    noise_mat = np.random.normal(
        mu_nnz * noise_mean_scale,
        std_nnz * noise_std_scale,
        (nnz_ids.sum(), num_col),
    )
    original_feat[nnz_ids] = (
        origin_p * original_feat[nnz_ids] + noise_p * noise_mat
    )
    # print(snapshot_id, total_snapshot, origin_p, noise_p)
    return original_feat
