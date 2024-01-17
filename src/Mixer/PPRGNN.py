from typing import List, Union, Dict, Optional
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# from torchsummary import summary

import functorch
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import normalize as sk_norm

from Mixer.PPRGNN_utils import (
    ModelPerfResult,
    get_hash_embed,
    get_hash_LUT,
    topk_norm,
    bottomk_norm,
    randomk_norm,
    simulate_additive_noise,
)


def poly_func(x, poly_coefs):
    out = torch.zeros_like(x)
    # skip offset
    for k in range(poly_coefs.shape[0] - 1):
        out += (x ** (k + 1)) * (poly_coefs[k + 1])
    return out


# split models


class VertexFormer(nn.Module):
    def __init__(
        self,
        rs: int,
        ppe_out_dim: int,
        node_raw_feat_dim: int,
        num_class: int,
        num_mlps: int,
        hidden_size_mlps: List[int],
        aggregate_type: str = "pprgo",
        drop_r: float = 0.15,
        mixrank_mode: str = "null",
        mixrank_num_samples: int = 32,
        polyrank_mode: str = "null",
        polyrank_order: int = 5,
    ) -> None:
        super(VertexFormer, self).__init__()

        self.rs = rs
        self.node_raw_feat_dim = node_raw_feat_dim
        self.ppe_out_dim = ppe_out_dim
        self.num_class = num_class
        self.num_mlps = num_mlps
        self.hidden_size_mlps = hidden_size_mlps
        self.first_hidden_size = (
            128  # self.hidden_size_mlps[0]  # = 1st hidden size
        )
        self.aggregate_type = aggregate_type
        self.drop_r = drop_r

        assert len(hidden_size_mlps) == num_mlps, (
            f"#mlp layers ({num_mlps}) != hidden_size_mlps"
            f" ({len(hidden_size_mlps)})"
        )
        assert self.aggregate_type in (
            "gcn",
            "mlp",
            "mlpppe",
            "ppe",
            "pprgo",
            "goppe",
            "gaussian",
            "mixrank",
            "polyrank",
        ), f"aggregate_type ({self.aggregate_type}) not included"

        if self.aggregate_type in ("gcn"):
            if self.aggregate_type == "gcn":
                self.conv1 = GCNConv(
                    self.node_raw_feat_dim, self.first_hidden_size
                )
                self.conv2 = GCNConv(self.first_hidden_size, self.num_class)

            else:
                raise NotImplementedError(
                    f"{self.aggregate_type} no implementation"
                )
        else:
            # customerize nn for merge different features.
            self.custom_nn(
                mixrank_mode,
                mixrank_num_samples,
                polyrank_mode,
                polyrank_order,
            )
            # this common linear layer
            self.dropout = nn.Dropout(p=self.drop_r)
            self.mlps = nn.ModuleList([])
            # self.bns = nn.ModuleList([])
            _last_h_size = self.first_hidden_size
            # self.feat_bn = nn.BatchNorm1d(_last_h_size)
            for i, _h_size in enumerate(hidden_size_mlps):
                self.mlps.append(nn.Linear(_last_h_size, _h_size))
                # self.bns.append(nn.BatchNorm1d(_h_size))
                _last_h_size = _h_size
            self.mlps.append(nn.Linear(_last_h_size, self.num_class))
            # do not add bn to output layer.

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        if self.aggregate_type in ("gcn"):
            self._init_weights(self.conv1)
            self._init_weights(self.conv2)
        else:
            self.mlps.apply(self._init_weights)
            # self.bns.apply(self._init_weights)
            # self._init_weights(self.feat_bn)

            if self.aggregate_type == "mlp":
                self._init_weights(self.mlp_mlp)
            elif self.aggregate_type == "ppe":
                self._init_weights(self.ppe_mlp)
            elif self.aggregate_type == "pprgo":
                self._init_weights(self.pprgo_mlp)
            elif self.aggregate_type == "goppe":
                self._init_weights(self.goppe_mlp_ppe)
                self._init_weights(self.goppe_mlp_feature)
                self._init_weights(self.goppe_merge_mlp)
                self._init_weights(self.goppe_merge_mlp_2)
                self.goppe_merge_coef.data.fill_(1.0)
                self.goppe_merge_coef[0] = 1.0
                self.goppe_merge_coef[1] = 1.0
            elif self.aggregate_type == "mlpppe":
                self._init_weights(self.mlpppe_mlp_ppe)
                self._init_weights(self.mlpppe_mlp_feature)
                self._init_weights(self.mlpppe_merge_mlp)
                self._init_weights(self.mlpppe_merge_mlp_2)
                self.mlpppe_merge_coef.data.fill_(1.0)
                self.mlpppe_merge_coef[0] = 1.0
                self.mlpppe_merge_coef[1] = 1.0
            elif self.aggregate_type == "gaussian":
                self._init_weights(self.gaussian_mlp)
            elif self.aggregate_type == "mixrank":
                self._init_weights(self.mixrank_mlp)
            elif self.aggregate_type == "polyrank":
                self._init_weights(self.polyrank_mlp)
                self.poly_coefs.data.fill_(0.0)
                self.poly_coefs[1] = 1.0
            else:
                raise NotImplementedError(
                    f"aggregate_type ({self.aggregate_type}) not implemented."
                )
        print("reinit weights")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.01)
        if isinstance(m, nn.BatchNorm1d):
            m.reset_running_stats()
        if isinstance(m, GCNConv):
            m.reset_parameters()
        # print(f"reinit weights {m}")

    def custom_nn(
        self, mixrank_mode, mixrank_num_samples, polyrank_mode, polyrank_order
    ):
        if self.aggregate_type == "mlp":
            self.mlp_mlp = nn.Linear(
                self.node_raw_feat_dim, self.first_hidden_size
            )
        elif self.aggregate_type == "ppe":
            self.ppe_mlp = nn.Linear(self.ppe_out_dim, self.first_hidden_size)
        elif self.aggregate_type == "pprgo":
            self.pprgo_mlp = nn.Linear(
                self.node_raw_feat_dim, self.first_hidden_size
            )
        elif self.aggregate_type == "goppe":
            # use ppe as positional encoding, additive or concat signal
            self.goppe_mlp_ppe = nn.Linear(
                self.ppe_out_dim, self.first_hidden_size
            )
            self.goppe_mlp_feature = nn.Linear(
                self.node_raw_feat_dim, self.first_hidden_size
            )
            self.goppe_merge_mlp = nn.Linear(
                int(self.first_hidden_size * 2), self.first_hidden_size
            )
            self.goppe_merge_mlp_2 = nn.Linear(
                int(self.first_hidden_size), self.first_hidden_size
            )
            self.goppe_merge_coef = nn.Parameter(
                torch.FloatTensor(torch.zeros(2))
            )
        elif self.aggregate_type == "mlpppe":
            self.mlpppe_mlp_ppe = nn.Linear(
                self.ppe_out_dim, self.first_hidden_size
            )
            self.mlpppe_mlp_feature = nn.Linear(
                self.node_raw_feat_dim, self.first_hidden_size
            )
            self.mlpppe_merge_mlp = nn.Linear(
                int(self.first_hidden_size * 2), self.first_hidden_size
            )
            self.mlpppe_merge_mlp_2 = nn.Linear(
                int(self.first_hidden_size), self.first_hidden_size
            )
            self.mlpppe_merge_coef = nn.Parameter(
                torch.FloatTensor(torch.zeros(2))
            )
        elif self.aggregate_type == "gaussian":
            self.gaussian_mlp = nn.Linear(
                self.node_raw_feat_dim, self.first_hidden_size
            )
        elif self.aggregate_type == "mixrank":
            self.mixrank_mode = mixrank_mode
            self.mixrank_num_samples = mixrank_num_samples
            self.mixrank_mlp = nn.Linear(
                self.node_raw_feat_dim, self.first_hidden_size
            )
        elif self.aggregate_type == "polyrank":
            self.polyrank_mode = polyrank_mode
            self.polyrank_order = polyrank_order
            # add para to model
            self.poly_coefs = nn.Parameter(
                torch.FloatTensor(torch.zeros(self.polyrank_order))
            )
            self.polyrank_mlp = nn.Linear(
                self.node_raw_feat_dim, self.first_hidden_size
            )
        else:
            raise NotImplementedError(
                f"aggregate_type ({self.aggregate_type}) not implemented."
            )

    def forward(
        self,
        x: torch.tensor,
        selected_index: Optional[torch.tensor] = None,
        edge_index: Optional[torch.tensor] = None,
    ):
        """Apply MLP over created feature."""
        if self.aggregate_type == "gcn":
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)
            return x[selected_index]

        elif self.aggregate_type == "mlp":
            x = self.mlp_mlp(x)
        elif self.aggregate_type == "ppe":
            x = self.ppe_mlp(x)
        elif self.aggregate_type == "pprgo":
            x = self.pprgo_mlp(x)
        elif self.aggregate_type == "goppe":
            # use ppe as positional encoding, additive signal
            hash_emb, ppv_feat_emb = x
            x_ppe = self.goppe_mlp_ppe(hash_emb)
            x_feat = self.goppe_mlp_feature(ppv_feat_emb)
            # self.goppe_merge_coef[1]
            # self.goppe_merge_coef[0]

            # option1: concat and down project
            # x_ppe = F.normalize(x_ppe)
            # x_feat = F.normalize(x_feat)
            x = torch.cat((x_ppe, x_feat), dim=1)
            x = self.goppe_merge_mlp(x)
            # x = self.goppe_merge_mlp_2(x)

            # option2: additie
            # x = x_ppe + x_feat
            # x = F.normalize(x_ppe) + F.normalize(x_feat)

            # option 3: learnable agg weights
            # x = torch.cat((x_ppe, x_feat), dim=1)
            # x = self.goppe_merge_mlp(x)
            # print(self.goppe_merge_coef)

            # option 4: just use feature.
            # x = x_feat
        elif self.aggregate_type == "mlpppe":
            hash_emb, feat_emb = x
            x_ppe = self.mlpppe_mlp_ppe(hash_emb)
            x_feat = self.mlpppe_mlp_feature(feat_emb)
            x = torch.cat((x_ppe, x_feat), dim=1)
            x = self.mlpppe_merge_mlp(x)
        elif self.aggregate_type == "gaussian":
            x = self.gaussian_mlp(x)
        elif self.aggregate_type == "mixrank":
            x = self.mixrank_mlp(x)
        elif self.aggregate_type == "polyrank":
            sliced_node_ids, sliced_ppv_csr_mat, full_feat_mat, hash_emb = x
            x = self.polyrank_forward(
                sliced_node_ids,
                sliced_ppv_csr_mat,
                full_feat_mat,
                hash_emb,
                self.polyrank_mode,
            )
        else:
            raise NotImplementedError(
                f"aggregate_type ({self.aggregate_type}) not implemented."
            )

        # print(self.aggregate_type, x.shape)

        # apply same before mlps
        # x = self.feat_bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        # common mlps for classification.
        for i in range(len(self.mlps)):
            x = self.mlps[i](x)
            if i == len(self.mlps) - 1:
                # do not apply these to last output layer.
                break
            # x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        # x = F.log_softmax(x, dim=1)
        return x

    def polyrank_forward(
        self,
        sliced_node_ids,
        sliced_ppv_csr_mat,
        full_feat_mat,
        hash_emb,
        polyrank_mode: str = "all",
    ):
        def __op(
            polyrank_order,
            sliced_node_ids,
            sliced_ppv_csr_mat,
            full_feat_mat,
            hash_emb,
            polyrank_mode: str = "linear",
        ):
            if polyrank_mode == "all":
                x = torch.sparse.mm(
                    sliced_ppv_csr_mat,
                    full_feat_mat,
                )
            elif polyrank_mode == "linear":
                x = self.poly_coefs[1] * torch.sparse.mm(
                    sliced_ppv_csr_mat,
                    full_feat_mat,
                )
            elif polyrank_mode == "softmax":
                # apply poly value to ppr data.
                # should be implemented in cpu mode.
                # no learnable param here
                sliced_ppv_coo_mat = sliced_ppv_csr_mat.to_sparse_coo()

                # print("before", sliced_ppv_coo_mat)
                # sliced_ppv_coo_mat_soft = torch.sparse.softmax(
                #     sliced_ppv_coo_mat,
                #     dim=1,
                # )
                # print("after", sliced_ppv_coo_mat_soft)
                # x = torch.sparse.mm(
                #     sliced_ppv_coo_mat_soft,
                #     full_feat_mat,
                # )

                sliced_ppv_dense_mat = sliced_ppv_coo_mat.to_dense()
                # print("before", sliced_ppv_dense_mat[0, :])
                sliced_ppv_dense_mat[sliced_ppv_dense_mat <= 1e-3] = -torch.inf
                # sliced_ppv_dense_mat[sliced_ppv_dense_mat == 0.0] = -torch.inf
                sliced_ppv_dense_mat_softmax = F.softmax(
                    sliced_ppv_dense_mat, dim=1
                )
                full_feat_mat_clean = torch.nan_to_num(full_feat_mat, nan=0.0)
                x = torch.mm(
                    sliced_ppv_dense_mat_softmax,
                    full_feat_mat_clean,
                )
                # print("after", sliced_ppv_dense_mat_softmax[0, :])

            elif polyrank_mode == "power":
                # apply poly value to ppr data.
                # does not work.
                sliced_ppv_coo_mat = sliced_ppv_csr_mat.to_sparse_coo()
                sliced_ppv_coo_mat_power = sliced_ppv_coo_mat.pow(2)
                x = torch.sparse.mm(
                    sliced_ppv_coo_mat_power,
                    full_feat_mat,
                )

            elif polyrank_mode == "poly":
                # start from dense version
                sliced_ppv_coo_mat = sliced_ppv_csr_mat.to_sparse_coo()
                full_feat_mat_clean = torch.nan_to_num(full_feat_mat, nan=0.0)

                # sliced_ppv_coo_mat_power = sliced_ppv_coo_mat.pow(2)
                # x = torch.sparse.mm(
                #     sliced_ppv_coo_mat_power,
                #     full_feat_mat,
                # )

                sliced_ppv_dense_mat = sliced_ppv_coo_mat.to_dense()
                zero_mask = sliced_ppv_dense_mat == 0.0
                # get polynomial fit
                sliced_ppv_dense_mat_poly = functorch.vmap(poly_func)(
                    sliced_ppv_dense_mat,
                    poly_coefs=self.poly_coefs,
                )
                # print("mean nnz:", zero_mask.sum(axis=1))
                sliced_ppv_dense_mat_poly[zero_mask] = -torch.inf
                sliced_ppv_dense_mat_poly = F.softmax(
                    sliced_ppv_dense_mat_poly, dim=1
                )

                # trick: torch mm will check type, cannot proceed with nan
                x = torch.mm(
                    sliced_ppv_dense_mat_poly,
                    full_feat_mat_clean,
                )

            return x

        x = __op(
            self.polyrank_order,
            sliced_node_ids,
            sliced_ppv_csr_mat,
            full_feat_mat,
            hash_emb,
            polyrank_mode=polyrank_mode,
        )

        x = self.polyrank_mlp(x)
        return x

    @torch.no_grad()
    def mixrank(
        self,
        sliced_node_ids: torch.tensor,
        sliced_ppv_csr_mat: torch.sparse_csr_tensor,  # sparse
        full_feat_mat: torch.tensor,
        hash_emb: torch.tensor,
        mixrank_mode: str = "high_low_ppr",
        mixrank_num_samples: int = 32,
    ):
        """art: use mix rank. mixing ppr and aggregate features"""
        if mixrank_mode == "random_sample":
            sliced_ppv_csr_mat_random = randomk_norm(
                max_rnd_k=mixrank_num_samples,
                matrix_input=sliced_ppv_csr_mat.to_dense().numpy(),
            ).to_sparse_csr()
            return torch.sparse.mm(
                sliced_ppv_csr_mat_random,
                full_feat_mat,
            )

        elif mixrank_mode == "high_ppr":
            sliced_ppv_csr_mat_high = topk_norm(
                max_top_k=mixrank_num_samples,
                matrix_input=sliced_ppv_csr_mat.to_dense().numpy(),
            ).to_sparse_csr()
            return torch.sparse.mm(
                sliced_ppv_csr_mat_high,
                full_feat_mat,
            )

        elif mixrank_mode == "low_ppr":
            sliced_ppv_csr_mat_low = bottomk_norm(
                max_btm_k=mixrank_num_samples,
                matrix_input=sliced_ppv_csr_mat.to_dense().numpy(),
            ).to_sparse_csr()
            return torch.sparse.mm(
                sliced_ppv_csr_mat_low,
                full_feat_mat,
            )

        elif mixrank_mode == "high_low_ppr":
            sliced_ppv_dense_mat_mix_high_low = topk_norm(
                max_top_k=mixrank_num_samples // 2,
                matrix_input=sliced_ppv_csr_mat.to_dense().numpy(),
            ) + bottomk_norm(
                max_btm_k=mixrank_num_samples // 2,
                matrix_input=sliced_ppv_csr_mat.to_dense().numpy(),
            )
            sliced_ppv_csr_mat_mix_high_low = (
                sliced_ppv_dense_mat_mix_high_low.to_sparse_csr()
            )
            return torch.sparse.mm(
                sliced_ppv_csr_mat_mix_high_low,
                full_feat_mat,
            )

        elif mixrank_mode == "all":
            return torch.sparse.mm(
                sliced_ppv_csr_mat,
                full_feat_mat,
            )

        else:
            raise NotImplementedError(
                f"mixrank_mode:{mixrank_mode} not implemented"
            )

    def former_feat(
        self,
        sliced_node_ids: np.ndarray,
        sliced_ppv_csr_mat: Union[sp.csr_matrix, torch.sparse_csr_tensor],
        full_feat_mat: Union[np.ndarray, torch.tensor],
        **kwargs,
    ):
        """
        This function creates features for different aggregation methods
        Note that:
        - Some methods (mlp/pprgo/mixrank) are para-free (no gradient).
        - For polyrank (learnable agg method), it returns a tuple of:
            (sliced_node_ids, sliced_ppv_csr_mat, full_feat_mat)
            We aggregate features on gradient tape to learn agg params.

        """

        aggregate_type = self.aggregate_type
        # return hash function of ppv (DynPPE)
        assert "ppe_hashcache_id" in kwargs, "ppe_hashcache_id?"
        assert "ppe_hash_cache_sign" in kwargs, "ppe_hash_cache_sign?"
        assert "ppe_out_dim" in kwargs, "ppe_out_dim?"
        if isinstance(sliced_ppv_csr_mat, torch.Tensor):
            indices = sliced_ppv_csr_mat.col_indices().numpy()
            indptr = sliced_ppv_csr_mat.crow_indices().numpy()
            data = sliced_ppv_csr_mat.values().numpy()
        else:
            indices = sliced_ppv_csr_mat.indices
            indptr = sliced_ppv_csr_mat.indptr
            data = sliced_ppv_csr_mat.data
        # numba speeded
        hash_emb: np.ndarray = get_hash_embed(
            kwargs["ppe_hashcache_id"],
            kwargs["ppe_hash_cache_sign"],
            kwargs["ppe_out_dim"],
            sliced_node_ids,
            sliced_ppv_csr_mat.shape[1],
            indices=indices,
            indptr=indptr,
            data=data,
        )
        # l2 normalized
        hash_emb: np.ndarray = sk_norm(hash_emb, norm="l2")

        if aggregate_type == "gcn":
            with torch.no_grad():
                return full_feat_mat
        elif aggregate_type == "mlpppe":
            with torch.no_grad():
                hash_emb = torch.tensor(hash_emb)
                select_feat_mat = full_feat_mat[sliced_node_ids, :]
                return (hash_emb, select_feat_mat)
        elif aggregate_type == "mlp":
            # return only node feature without aggregation
            with torch.no_grad():
                return full_feat_mat[sliced_node_ids, :]

        elif aggregate_type == "ppe":
            with torch.no_grad():
                if isinstance(sliced_ppv_csr_mat, torch.Tensor):
                    return torch.tensor(hash_emb)
                return hash_emb

        elif aggregate_type == "pprgo":
            with torch.no_grad():
                # return ppr weighted node features
                assert "pprgo_topk" in kwargs, "pprgo_topk?"
                sliced_ppv_csr_mat_high = topk_norm(
                    max_top_k=kwargs["pprgo_topk"],
                    matrix_input=sliced_ppv_csr_mat.to_dense().numpy(),
                ).to_sparse_csr()

                return torch.sparse.mm(
                    sliced_ppv_csr_mat_high,
                    full_feat_mat,
                )

        elif aggregate_type == "goppe":
            # mix pprgo and ppe
            # return hash embedding and ppv-weighted embeddings
            with torch.no_grad():
                if isinstance(sliced_ppv_csr_mat, torch.Tensor):
                    hash_emb = torch.tensor(hash_emb)
                    ppv_feat_emb = torch.sparse.mm(
                        sliced_ppv_csr_mat,
                        full_feat_mat,
                    )
                    # goppe_emb = torch.hstack((hash_emb, ppv_feat_emb))
                    goppe_emb = (hash_emb, ppv_feat_emb)
                else:
                    ppv_feat_emb = sliced_ppv_csr_mat.dot(full_feat_mat)
                    # goppe_emb = np.hstack((hash_emb, ppv_feat_emb))
                    goppe_emb = (hash_emb, ppv_feat_emb)

            return goppe_emb

        elif aggregate_type == "mixrank":
            # mixrank: aggregate features based on ppr in a
            # parameter-free manner.
            with torch.no_grad():
                if isinstance(sliced_ppv_csr_mat, torch.Tensor):
                    hash_emb: torch.tensor = torch.tensor(hash_emb)
                    mixrank_emb = self.mixrank(
                        sliced_node_ids,
                        sliced_ppv_csr_mat,
                        full_feat_mat,
                        hash_emb,
                        self.mixrank_mode,
                        self.mixrank_num_samples,
                    )
                else:
                    raise NotImplementedError(
                        "MixRank must use torch.sparse backend."
                        "use the flag --use_torch_sparse"
                    )
                return mixrank_emb

        elif aggregate_type == "polyrank":
            # polyrank: learn a polynomial function to modify ppr-based
            # aggregation weight.
            # It is learnable, feature agg was tracked by gradient tape.
            if isinstance(sliced_ppv_csr_mat, torch.Tensor):
                # construct hash_emb as leaf tensors
                hash_emb: torch.tensor = torch.tensor(
                    hash_emb,
                    requires_grad=False,
                )
                sliced_node_ids: torch.tensor = torch.tensor(
                    sliced_node_ids,
                    requires_grad=False,
                )
            else:
                raise NotImplementedError(
                    "Polyrank must use torch.sparse backend."
                    "use the flag --use_torch_sparse"
                )
            return (
                sliced_node_ids,
                sliced_ppv_csr_mat,
                full_feat_mat,
                hash_emb,
            )

        elif aggregate_type == "gaussian":
            # generate random embeddings as monkey baseline.
            input_shape = full_feat_mat[sliced_node_ids, :].shape
            if isinstance(sliced_ppv_csr_mat, torch.Tensor):
                return torch.normal(0, 0.1, size=input_shape)
            else:  # isinstance(sliced_ppv_csr_mat, sp.csr_matrix):
                return np.random.normal(0, 0.1, input_shape)

        else:
            raise NotImplementedError(
                f"aggregate_type:{aggregate_type} is not implemented"
            )


@torch.no_grad()
def test_model(
    model: VertexFormer,
    X: torch.tensor,
    Y_truth: torch.tensor,
    loss_func,
    full_feat_mat: Optional[torch.tensor] = None,
    edge_index: Optional[torch.tensor] = None,
    selected_node_ids: Optional[torch.tensor] = None,
) -> ModelPerfResult:
    """test and record model performance"""
    model.eval()

    if model.aggregate_type in ("gcn"):
        # print("edge_index", edge_index.shape, type(edge_index))
        # _ = input("test_model1")
        Y_pred_logits = model(full_feat_mat, selected_node_ids, edge_index)
    else:
        Y_pred_logits = model(X)

    loss_val = loss_func(Y_pred_logits, Y_truth)

    Y_pred = F.softmax(Y_pred_logits, dim=1)
    Y_pred = Y_pred.detach().cpu().numpy()
    Y_truth = Y_truth.detach().cpu().numpy()
    loss_val = loss_val.detach().cpu().numpy()
    # print(Y_truth, Y_pred)
    # _ = input("test_model1")
    roc_auc = roc_auc_score(Y_truth, Y_pred)
    clf_report_dict: Dict = classification_report(
        np.argmax(Y_truth, axis=1), np.argmax(Y_pred, axis=1), output_dict=True
    )

    return ModelPerfResult(
        loss_val,
        roc_auc,
        clf_report_dict,
    )


def get_model_helper(
    total_num_nodes: int,
    node_raw_feat_dim: int,
    ppe_out_dim: int,
    num_class: int,
    num_mlps: int,
    hidden_size_mlps: List[int],
    aggregate_type: str,
    rs: int,
    is_cuda_used: bool = False,
    device: str = "cuda",
    drop_r: float = 0.15,
    mixrank_mode: str = "null",
    mixrank_num_samples: int = 32,
    polyrank_mode: str = "null",
    polyrank_order: int = 5,
):
    """Get torch model and any model required data"""

    assert num_mlps == len(
        hidden_size_mlps
    ), f"num_mlps:{num_mlps}!=len(hidden_size_mlps):{len(hidden_size_mlps)}"

    ppe_node_id_2_dim_id = np.array([], dtype=np.int32)
    ppe_node_id_2_sign = np.array([], dtype=np.int8)
    # if aggregate_type in ("ppe", "goppe", "mixrank", "polyrank"):
    ppe_node_id_2_dim_id, ppe_node_id_2_sign = get_hash_LUT(
        total_num_nodes,
        ppe_out_dim,
        rnd_seed=rs,
    )

    model = VertexFormer(
        rs=rs,
        ppe_out_dim=ppe_out_dim,
        node_raw_feat_dim=node_raw_feat_dim,
        num_class=num_class,
        num_mlps=num_mlps,
        hidden_size_mlps=hidden_size_mlps,
        aggregate_type=aggregate_type,
        drop_r=drop_r,
        mixrank_mode=mixrank_mode,
        mixrank_num_samples=mixrank_num_samples,
        polyrank_mode=polyrank_mode,
        polyrank_order=polyrank_order,
    )

    if is_cuda_used:
        model.to(device)

    return model, ppe_node_id_2_dim_id, ppe_node_id_2_sign


def prepare_model_input(
    snapshot_id,
    total_snapshots,
    model,
    input_node_ids,
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
):
    # snapshot_id, total_snapshots,
    if use_simulated_noise:
        full_feat_mat = simulate_additive_noise(
            full_feat_mat,
            snapshot_id,
            total_snapshots,
        )
        print("simulate gaussian noise to node feature")

    # retrive PPVs from ppr_estimator
    sliced_ppv_dense_mat = np.array(
        [ppr_estimator.dict_p_arr[i] for i in input_node_ids],
    ).astype(np.float64)

    if is_torch_sparse:
        sliced_ppv_csr_mat = torch.tensor(
            sliced_ppv_dense_mat, dtype=torch.float64
        ).to_sparse_csr()
        full_feat_mat = torch.tensor(
            full_feat_mat,
            dtype=torch.float64,
        )
    else:
        sliced_ppv_csr_mat = sp.csr_matrix(sliced_ppv_dense_mat)
        # nnz = sliced_ppv_csr_mat.nnz

    # prepare feature accoroding different aggregate_type
    former_X = model.former_feat(
        input_node_ids,  # sliced_node_ids
        sliced_ppv_csr_mat,  # sliced_node_ids
        full_feat_mat,  # full feature mat (already loaded in memory).
        ppe_hashcache_id=ppe_hashcache_id,
        ppe_hash_cache_sign=ppe_hash_cache_sign,
        ppe_out_dim=ppe_out_dim,
        **kwargs,
    )
    target_Y: np.ndarray = full_lb_mat[input_node_ids]  # bool mat
    target_Y = torch.tensor(target_Y, dtype=torch.float)
    assert is_cuda_used, f"not used cuda, is_cuda_used = {is_cuda_used}"
    # if is_cuda_used:
    target_Y = target_Y.to(device)

    if model.aggregate_type == "polyrank":
        # polyrank needs aggregate features on gradient tape
        # move the feature agg process into model.forward()
        (
            sliced_node_ids,
            sliced_ppv_csr_mat,
            full_feat_mat,
            hash_emb,
        ) = former_X
        sliced_node_ids = sliced_node_ids.to(device)
        sliced_ppv_csr_mat = sliced_ppv_csr_mat.to(device)
        full_feat_mat = full_feat_mat.to(device)
        hash_emb = hash_emb.to(device)
        return (
            (sliced_node_ids, sliced_ppv_csr_mat, full_feat_mat, hash_emb),
            target_Y,
            full_feat_mat,
        )
    elif model.aggregate_type in ("goppe", "mlpppe"):
        hash_emb, ppv_feat_emb = former_X
        ppv_feat_emb = ppv_feat_emb.to(device, dtype=torch.float)
        hash_emb = hash_emb.to(device, dtype=torch.float)
        return ((hash_emb, ppv_feat_emb), target_Y, full_feat_mat)
    else:
        if not isinstance(former_X, torch.Tensor):
            former_X = torch.tensor(former_X, dtype=torch.float)
        else:
            former_X = former_X.clone().detach().float()
        former_X = former_X.to(device)

        # print("former_X.shape", former_X.shape)
        return former_X, target_Y, full_feat_mat
