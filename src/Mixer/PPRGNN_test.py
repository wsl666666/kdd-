import argparse
import itertools
import random
import sys
import unittest
import warnings
import pandas as pd
import numpy as np
import torch

warnings.filterwarnings("ignore")

from time import perf_counter as p_time

sys.path.insert(0, "/home/xingzguo/git_project/DynMixer/src")
sys.path.insert(0, "/home/xingzguo/git_project/DynMixer/src/DynGL")
sys.path.insert(0, "/home/xingzguo/git_project/DynMixer/src/DynGLReaders")

import scipy.sparse as sp

from DynGL.DynGraph.DynGraph import DynGraph
from DynGL.DynGraph.DynGraphLoader import DynGraphMetadata, DynGraphReader
from DynGL.DynPageRank import DynPPRAlgos
from DynGLReaders.DynGraphReaders import DynGraphReaderKarate
from Mixer.PPRGNN import VertexFormer
from Mixer.PPRGNN_utils import get_hash_LUT, split_train_dev_test, setup_seed


class test_integrate_pprgnn(unittest.TestCase):
    def skip_test_vertexformer_former_feature_creator(self):
        """
        benchmarking the speed of vertex former's feature creator:
            - mlp: only use node feature itself (tensor slicing speed)
            - ppe: only use hash ppv embeddings (hash function speed)
            - pprgo: use ppv as weight to aggreagate node featre
                (sparse_ppv_mat * dense_node_feature_mat, test spMD)
            - goppe: use both ppe and pprgo feature

        Note:
            - METHODS   --torch--vs--scipy--
            - mlp:      0.000040 vs 0.000007
            - ppe:      0.000830 vs 0.000735
            - pprgo:    0.008170 vs 0.026841
            - goppe:    0.009310 vs 0.029020
        Takeways:
            - scipy ndarray slicing is faster than torch (see mlp).
            - torch.sparse.mm (sparse_mat * dense_mat) is 3-5 times
                faster than scipy.sparse (see pprgo).
        """
        # whether to use torch sparse backend
        use_torch_sparse = True  # True False
        benchmark_repeat_time = 100
        rs = 10
        r_state = np.random.RandomState(rs)
        setup_seed(rs)

        total_num_nodes = 1000000
        nnz_per_row = 3000  # spDM computation bottleneck.
        dim_tag_node_feat = 300

        # aggregate_type = "mlp" # constant
        # aggregate_type = "ppe"  # complexiy depends on sparsity
        # aggregate_type = "pprgo"  # complexiy depends on sparsity
        aggregate_type = "goppe"  # complexiy depends on sparsity

        bs_size = 32  # 32
        input_dim = 128
        num_class = 5
        num_mlps = 0
        hidden_size_mlps = []

        benchmark_records = []

        for total_num_nodes in [100000, 1000000]:
            for use_torch_sparse in [True, False]:
                for aggregate_type in ["mlp", "ppe", "pprgo", "goppe"]:
                    (
                        time_mean,
                        time_std,
                        nnz_mean,
                        nnz_std,
                    ) = self.run_speed_test_helper(
                        use_torch_sparse,
                        benchmark_repeat_time,
                        rs,
                        r_state,
                        total_num_nodes,
                        nnz_per_row,
                        dim_tag_node_feat,
                        aggregate_type,
                        bs_size,
                        input_dim,
                        num_class,
                        num_mlps,
                        hidden_size_mlps,
                    )
                    _records = {
                        "aggregate_type": aggregate_type,
                        "use_torch_sparse": use_torch_sparse,
                        "total_num_nodes": total_num_nodes,
                        "nnz_per_row": nnz_per_row,
                        "dim_tag_node_feat": dim_tag_node_feat,
                        "time_mean": time_mean,
                        "time_std": time_std,
                        "nnz_mean": nnz_mean,
                        "nnz_std": nnz_std,
                        "rs": rs,
                        "benchmark_repeat_time": benchmark_repeat_time,
                        "bs_size": bs_size,
                        "input_dim": input_dim,
                        "num_class": num_class,
                        "num_mlps": num_mlps,
                    }
                    print(_records)
                    benchmark_records.append(_records)
        df_speed_benchmark = pd.DataFrame.from_records(benchmark_records)
        df_speed_benchmark.to_json("./benchmark_stats/df_speed_benchmark.json")

    def skip_test_sparse_benchmark(self):
        use_torch_sparse = True  # True False
        benchmark_repeat_time = 500
        total_num_nodes = 50000
        nnz_per_row = 3000
        dim_tag_node_feat = 300
        bs_size = 128
        rs = 10
        r_state = np.random.RandomState(rs)
        setup_seed(rs)

        # random feature
        full_feat_mat = r_state.random(
            (total_num_nodes, dim_tag_node_feat)
        ).astype(np.float32)
        select_node_id_training = r_state.randint(
            0, total_num_nodes, (1, bs_size)
        ).astype(np.int32)
        sliced_ppv_csr_mat = self.fake_ppv_mat(
            total_num_nodes,
            select_node_id_training,
            nnz_per_row,
        )
        if use_torch_sparse:
            full_feat_mat = torch.tensor(full_feat_mat)
            select_node_id_training = torch.tensor(
                select_node_id_training,
                dtype=torch.long,
            )
            sliced_ppv_coo_mat = torch.Tensor(
                sliced_ppv_csr_mat,
            ).to_sparse()  # coo
            sliced_ppv_csr_mat = torch.Tensor(
                sliced_ppv_csr_mat
            ).to_sparse_csr()

        else:
            # use scipy sparse matrix
            sliced_ppv_csr_mat = sp.csr_matrix(sliced_ppv_csr_mat)

        if not use_torch_sparse:
            # benchmark 1: use scipy sparse multiplication
            # avg-torch-sparse:
            # mean-per-loop: 0.009268946199445054
            # std-per-loop: 0.0002690708237289303
            perf_counter = []
            for _ in range(benchmark_repeat_time):
                _t = p_time()
                _ = self.scipy_spDM(
                    full_feat_mat,
                    sliced_ppv_csr_mat,
                )
                perf_counter.append(p_time() - _t)
            print(
                "avg-scipy:",
                np.mean(perf_counter),
                np.std(perf_counter),
            )

        if use_torch_sparse:
            # benchmark : use sliced matrices for calculation
            # the performance depends on the torch's spDM()
            # avg-torch-sparse:
            # mean-per-loop: 0.006591805458301678
            # std-per-loop: 0.0004334492152097655

            perf_counter = []
            for _ in range(benchmark_repeat_time):
                _t = p_time()
                _ = self.torch_spDM(
                    full_feat_mat,
                    sliced_ppv_csr_mat,
                )
                perf_counter.append(p_time() - _t)
            print(
                "avg-torch-sparse:",
                np.mean(perf_counter),
                np.std(perf_counter),
            )

            # benchmark : use sliced matrices for calculation
            # it has slicing overhead
            # avg-torch-sparse:
            # mean-per-loop: 0.04057011612458154
            # std-per-loop: 0.0031508531445460074
            perf_counter = []
            for _ in range(benchmark_repeat_time):
                _t = p_time()
                _ = self.complex_ppv_agg(
                    full_feat_mat,
                    sliced_ppv_coo_mat,
                )
                perf_counter.append(p_time() - _t)
            print(
                "avg-torch-sparse-sliced:",
                np.mean(perf_counter),
                np.std(perf_counter),
            )

            perf_counter = []
            for _ in range(benchmark_repeat_time):
                _t = p_time()
                _ = self.torch_scatter(
                    full_feat_mat,
                    sliced_ppv_coo_mat,
                )
                perf_counter.append(p_time() - _t)
            print(
                "avg-torch-scatter:",
                np.mean(perf_counter),
                np.std(perf_counter),
            )

    def run_speed_test_helper(
        self,
        use_torch_sparse,
        benchmark_repeat_time,
        rs,
        r_state,
        total_num_nodes,
        nnz_per_row,
        dim_tag_node_feat,
        aggregate_type,
        bs_size,
        input_dim,
        num_class,
        num_mlps,
        hidden_size_mlps,
    ):
        model = VertexFormer(
            rs=rs,
            mlp_input_dim=input_dim,
            num_class=num_class,
            num_mlps=num_mlps,
            hidden_size_mlps=hidden_size_mlps,
            aggregate_type=aggregate_type,
        )
        # print(model)

        # create fake whole node features matrix
        full_feat_mat = r_state.random(
            (total_num_nodes, dim_tag_node_feat)
        ).astype(np.float32)
        if use_torch_sparse:
            full_feat_mat = torch.tensor(full_feat_mat)

        # create fake training batch of node ids
        node_id_batch = r_state.randint(
            0, total_num_nodes, (5, bs_size)
        ).astype(np.int32)

        ppe_node_id_2_dim_id = None
        ppe_node_id_2_sign = None
        if model.aggregate_type in ("ppe", "goppe"):
            ppe_node_id_2_dim_id, ppe_node_id_2_sign = get_hash_LUT(
                total_num_nodes,
                input_dim,
                rnd_seed=rs,
            )

        mean_time = []
        total_nnz = []

        for batch_id in range(node_id_batch.shape[0]):
            select_node_id_training = node_id_batch[batch_id, :]
            sliced_ppv_csr_mat = self.fake_ppv_mat(
                total_num_nodes,
                select_node_id_training,
                nnz_per_row,
            )

            if use_torch_sparse:
                # node_id_batch = torch.tensor(node_id_batch, dtype=torch.long)
                sliced_ppv_csr_mat = torch.Tensor(
                    sliced_ppv_csr_mat
                ).to_sparse_csr()
                nnz = sliced_ppv_csr_mat.values().numpy().shape[0]
            else:
                sliced_ppv_csr_mat = sp.csr_matrix(sliced_ppv_csr_mat)
                nnz = sliced_ppv_csr_mat.nnz
            total_nnz.append(nnz)

            # aggregate_type
            for _ in range(benchmark_repeat_time):
                _t = p_time()
                former_x = VertexFormer.former_feat(
                    model.aggregate_type,
                    select_node_id_training,
                    sliced_ppv_csr_mat,
                    full_feat_mat,
                    ppe_hashcache_id=ppe_node_id_2_dim_id,
                    ppe_hash_cache_sign=ppe_node_id_2_sign,
                    ppe_out_dim=input_dim,
                )
                mean_time.append(p_time() - _t)
            # former_x = torch.tensor(former_x, dtype=torch.float)
            # y = model(former_x)
            # print(y.shape)
        return (
            np.mean(mean_time),
            np.std(mean_time),
            np.mean(total_nnz),
            np.std(total_nnz),
        )

    def fake_ppv_mat(
        self,
        total_num_nodes,
        select_node_id_training,
        nnz_per_row: int = 3000,
    ):
        sp_ppv = torch.zeros(
            (
                select_node_id_training.shape[0],
                total_num_nodes,
            )
        ).normal_(0, 0.1)

        # num_rows = select_node_id_training.shape[0]
        zero_rate = 1.0 - (1.0 * nnz_per_row / total_num_nodes)
        sliced_ppv_csr_mat = torch.nn.functional.dropout(
            sp_ppv,
            zero_rate,
        ).numpy()
        sliced_ppv_csr_mat = np.abs(sliced_ppv_csr_mat)
        row_sum = sliced_ppv_csr_mat.sum(axis=1).reshape(-1, 1)
        sliced_ppv_csr_mat = sliced_ppv_csr_mat / row_sum
        return sliced_ppv_csr_mat

    def scipy_spDM(self, feat_mat_all, ppv_mat):
        return ppv_mat.dot(feat_mat_all)

    def torch_spDM(self, feat_mat_all, ppv_mat):
        return torch.sparse.mm(ppv_mat, feat_mat_all)

    def complex_ppv_agg(
        self,
        feat_mat_all,
        ppv_mat: torch.sparse_coo_tensor,
    ):
        # slow
        nnz_ppv_indices = ppv_mat.indices()
        # 1: select nnz ppv values
        nnz_ppv_vals = ppv_mat.values()
        # 2: slice raw features based on nnz in ppv
        nnz_node_feats = feat_mat_all[nnz_ppv_indices[1]]

        bs = ppv_mat.shape[0]
        k = nnz_ppv_vals.shape[0]

        # 3: apply ppv to each rows using diagnal mat
        nnz_ppv_vals_sparse_diag = torch.sparse.spdiags(
            diagonals=nnz_ppv_vals,
            shape=(k, k),
            offsets=torch.LongTensor([0]),
        )
        nnz_node_feats_w_ppv = torch.sparse.mm(
            nnz_ppv_vals_sparse_diag,
            nnz_node_feats,
        )

        # 4. create sparse mat to aggregation weighted rows.
        coo_1 = nnz_ppv_indices[0]
        coo_2 = torch.arange(k)
        s = torch.sparse_coo_tensor(
            torch.vstack([coo_1, coo_2]),
            torch.ones(k),
            (bs, k),
        )
        return torch.sparse.mm(s, nnz_node_feats_w_ppv)

    def skip_test_split_train_dev_test_fix_all_train_dev_test(self):
        rs = 621
        curt_node_num = 10
        num_node_class = 3
        max_node_num = 100
        train_per_lb = 2
        dev_num_nodes = 2
        test_num_nodes = 2

        graph_curt_all_nodes_ids = np.arange(curt_node_num, dtype=np.uint32)
        graph_curt_all_nodes_lbs = np.zeros(
            (max_node_num, num_node_class),
            dtype=bool,
        )

        train_node_ids = np.array([], dtype=np.uint32)
        dev_node_ids = np.array([], dtype=np.uint32)
        test_node_ids = np.array([], dtype=np.uint32)

        data_strategy = "fix-all"
        graph_curt_all_nodes_lbs[0, 2] = True
        graph_curt_all_nodes_lbs[1, 0] = True
        graph_curt_all_nodes_lbs[2, 1] = True
        graph_curt_all_nodes_lbs[3, 2] = True
        graph_curt_all_nodes_lbs[4, 2] = True
        graph_curt_all_nodes_lbs[5, 2] = True
        graph_curt_all_nodes_lbs[6, 0] = True
        graph_curt_all_nodes_lbs[7, 1] = True
        graph_curt_all_nodes_lbs[8, 2] = True
        graph_curt_all_nodes_lbs[9, 2] = True

        snapshot_id = 0
        train_idx, dev_idx, test_idx = split_train_dev_test(
            graph_curt_all_nodes_ids,
            graph_curt_all_nodes_lbs,
            train_node_ids,
            dev_node_ids,
            test_node_ids,
            snapshot_id,
            rs,
            data_strategy,
            train_per_lb,
            dev_num_nodes,
            test_num_nodes,
        )
        print(
            f"{snapshot_id}\n "
            f"Train:{train_idx}\n"
            f"Dev:{dev_idx}\n"
            f"Test:{test_idx}"
        )

    def test_split_train_dev_test_fix_all_train_dev_test(self):
        rs = 621
        curt_node_num = 10
        num_node_class = 3
        max_node_num = 100
        train_per_lb = 1
        dev_num_nodes = 2
        test_num_nodes = 2

        graph_curt_all_nodes_ids = np.arange(curt_node_num, dtype=np.uint32)
        graph_curt_all_nodes_lbs = np.zeros(
            (max_node_num, num_node_class),
            dtype=bool,
        )

        train_node_ids = np.array([], dtype=np.uint32)
        dev_node_ids = np.array([], dtype=np.uint32)
        test_node_ids = np.array([], dtype=np.uint32)

        data_strategy = "add-test"
        graph_curt_all_nodes_lbs[0, 2] = True
        graph_curt_all_nodes_lbs[1, 0] = True
        graph_curt_all_nodes_lbs[2, 1] = True
        graph_curt_all_nodes_lbs[3, 2] = True
        graph_curt_all_nodes_lbs[4, 2] = True
        graph_curt_all_nodes_lbs[5, 2] = True
        graph_curt_all_nodes_lbs[6, 0] = True
        graph_curt_all_nodes_lbs[7, 1] = True
        graph_curt_all_nodes_lbs[8, 2] = True
        graph_curt_all_nodes_lbs[9, 2] = True

        snapshot_id = 0
        train_node_ids, dev_node_ids, test_node_ids = split_train_dev_test(
            graph_curt_all_nodes_ids,
            graph_curt_all_nodes_lbs,
            train_node_ids,
            dev_node_ids,
            test_node_ids,
            snapshot_id,
            rs,
            data_strategy,
            train_per_lb,
            dev_num_nodes,
            test_num_nodes,
        )
        print(
            f"-----snapshot: {snapshot_id}\n "
            f"Train:{train_node_ids}\n"
            f"Dev:{dev_node_ids}\n"
            f"Test:{test_node_ids}"
        )

        snapshot_id += 1
        test_num_nodes += 1
        train_node_ids, dev_node_ids, test_node_ids = split_train_dev_test(
            graph_curt_all_nodes_ids,
            graph_curt_all_nodes_lbs,
            train_node_ids,
            dev_node_ids,
            test_node_ids,
            snapshot_id,
            rs,
            data_strategy,
            train_per_lb,
            dev_num_nodes,
            test_num_nodes,
        )
        print(
            f"-----snapshot: {snapshot_id}\n "
            f"Train:{train_node_ids}\n"
            f"Dev:{dev_node_ids}\n"
            f"Test:{test_node_ids}"
        )

        snapshot_id += 1
        test_num_nodes += 1
        train_node_ids, dev_node_ids, test_node_ids = split_train_dev_test(
            graph_curt_all_nodes_ids,
            graph_curt_all_nodes_lbs,
            train_node_ids,
            dev_node_ids,
            test_node_ids,
            snapshot_id,
            rs,
            data_strategy,
            train_per_lb,
            dev_num_nodes,
            test_num_nodes,
        )
        print(
            f"-----snapshot: {snapshot_id}\n "
            f"Train:{train_node_ids}\n"
            f"Dev:{dev_node_ids}\n"
            f"Test:{test_node_ids}"
        )


if __name__ == "__main__":
    unittest.main()
