import numpy as np
from DynGL.DynGraph.DynGraphLoader import DynGraphReader
import os
from os.path import join as os_join
from os.path import exists as os_exists
from typing import Tuple, List
import json
from overrides import override
import pickle
import torch_geometric.transforms as T
import torch_geometric
from ogb.nodeproppred import NodePropPredDataset
import torch
import os
from typing import Any, Callable, List, Optional, Tuple
from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)

from typing import Any, Callable, Optional, Tuple


class EllipticBitcoinTempDataset(InMemoryDataset):
    r"""The Elliptic Bitcoin dataset of Bitcoin transactions from the
    `"Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional
    Networks for Financial Forensics" <https://arxiv.org/abs/1908.02591>`_
    paper.


        * - #nodes
          - #edges
          - #features
          - #classes
        * - 203,769
          - 234,355
          - 165
          - 2
    """
    url = "https://data.pyg.org/datasets/elliptic"

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return [
            "elliptic_txs_features.csv",
            "elliptic_txs_edgelist.csv",
            "elliptic_txs_classes.csv",
        ]

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self):
        for file_name in self.raw_file_names:
            path = download_url(f"{self.url}/{file_name}.zip", self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.remove(path)

    def _process_df(
        self, feat_df: Any, edge_df: Any, class_df: Any
    ) -> Tuple[Any, Any, Any]:
        print("test", feat_df)

        return feat_df, edge_df, class_df

    def process(self):
        import pandas as pd

        feat_df = pd.read_csv(self.raw_paths[0], header=None)
        edge_df = pd.read_csv(self.raw_paths[1])
        class_df = pd.read_csv(self.raw_paths[2])

        columns = {0: "txId", 1: "time_step"}
        feat_df = feat_df.rename(columns=columns)

        feat_df, edge_df, class_df = self._process_df(
            feat_df,
            edge_df,
            class_df,
        )

        x = torch.from_numpy(feat_df.loc[:, 2:].values).to(torch.float)

        # There exists 3 different classes in the dataset:
        # 0=licit,  1=illicit, 2=unknown
        mapping = {"unknown": 2, "1": 1, "2": 0}
        class_df["class"] = class_df["class"].map(mapping)
        y = torch.from_numpy(class_df["class"].values)

        mapping = {idx: i for i, idx in enumerate(feat_df["txId"].values)}
        edge_df["txId1"] = edge_df["txId1"].map(mapping)
        edge_df["txId2"] = edge_df["txId2"].map(mapping)
        # edge_df should sort
        edge_index = torch.from_numpy(edge_df.values).t().contiguous()

        # Timestamp based split:
        # train_mask: 1 - 34 time_step, test_mask: 35-49 time_step
        time_step = torch.from_numpy(feat_df["time_step"].values)
        train_mask = (time_step < 35) & (y != 2)
        test_mask = (time_step >= 35) & (y != 2)

        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            test_mask=test_mask,
        )

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    @property
    def num_classes(self) -> int:
        return 2


class DynGraphReaderKarate(DynGraphReader):
    def __init__(
        self,
        graph_dataset_name: str,
        local_dataset_dir_abs_path: str,
        graph_event_prefix: str = "snapshot-",
        verbose: bool = False,
    ):
        super().__init__(
            graph_dataset_name=graph_dataset_name,
            local_dataset_dir_abs_path=local_dataset_dir_abs_path,
            graph_event_prefix=graph_event_prefix,
            verbose=verbose,
        )

    @override
    def download_raw_data(self):
        """download data to local path

        save to self.node_feat_event_path

        """
        raw_data_file_path = os_join(self.raw_file_path, "edge_list.json")
        if os.path.exists(raw_data_file_path):
            print(f"file exists: {raw_data_file_path}")
        else:
            raise NotImplementedError("download karate")

    @override
    def parse_edge_event(self, out_buffer_size: int = 10000):
        """parse downloaded raw graph data to edge list, including edge
        timestamp, better to use streaming I/O manner.
        [
            [u: str, v:str, delta_w: float, t:float],
            ...
        ]
        edge_list_file_path = os_join(self.edge_event_path, "data.json")
        """

        raw_data_file_path = os_join(self.raw_file_path, "edge_list.json")
        edge_list_file_path = os_join(self.edge_event_path, "data.json")

        def __write_out(buffer, f_write):
            f_write.write("\n".join([json.dumps(_) for _ in buffer]))
            f_write.write("\n")

        if os.path.exists(edge_list_file_path):
            print(f"unsorted edge list already exists: {edge_list_file_path}")
        else:
            if not os.path.exists(self.edge_event_path):
                if self.verbose:
                    print(
                        f"edge event folder is created: {self.edge_event_path}"
                    )
                os.makedirs(self.edge_event_path)

            out_buffer = []
            with open(edge_list_file_path, "w", encoding="utf-8") as f_write:
                with open(raw_data_file_path, "r", encoding="utf-8") as f_read:
                    for line in f_read:
                        u, v, w, t = json.loads(line.strip())
                        u = str(u)
                        v = str(v)
                        w = float(w)
                        t = float(t)
                        out_buffer.append((u, v, w, t))

                        if len(out_buffer) == out_buffer_size:
                            __write_out(out_buffer, f_write)
                            out_buffer = []

                    if len(out_buffer) > 0:
                        __write_out(out_buffer, f_write)
                        out_buffer = []
            if self.verbose:
                print(f"edge struct event generated at {edge_list_file_path})")
        return None

    @override
    def parse_node_feature_event(self):
        """convert node feature event (e.g., assign/modify) to list,
        including its timestamp.
        [[ u:str, feature:np.ndarray, t:float ], ...]

        For example: save it to
        node_feat_event_file_path = os_join(
            self.node_feat_event_path, "node_feature_event.pkl"
        )

        """
        node_feat_event_file_path = os_join(
            self.node_feat_event_path, "node_feature_event.pkl"
        )
        if os.path.exists(node_feat_event_file_path):
            return None

        if not os.path.exists(self.node_feat_event_path):
            os.makedirs(self.node_feat_event_path)

        # for Karate: fake some
        fk_dim = 32
        total_node_feat_events = 100

        edge_list_file_path = os_join(self.edge_event_path, "data.json")
        nodes = []
        feats = []
        ts = []
        with open(edge_list_file_path, "r") as f:
            for i, line in enumerate(f):
                u, v, w, t = json.loads(line.strip())
                u = str(u)
                v = str(v)
                w = float(w)
                t = float(t)
                nodes.append(u)
                feats.append(np.random.normal(0, 5, fk_dim))
                ts.append(t)
                if len(nodes) == total_node_feat_events:
                    break
        fake_feat_events = []

        for i in range(len(nodes)):
            _u = str(nodes[i])
            _feat = feats[i].astype(np.float32)
            _t = float(ts[i])
            fake_feat_events.append((_u, _feat, _t))

        with open(node_feat_event_file_path, "wb") as f:
            pickle.dump(fake_feat_events, f)

        print(f"write to {node_feat_event_file_path}")

    @override
    def parse_node_label_event(self):
        """convert node label event (e.g., assign/modify) to list,
        including its timestamp.
        [[ u:str, label:np.array[Bool], t:float ], ...]

        encode node label into array of int (assuming multi-label
        classification).

        For example: save to
        node_label_event_file_path = os_join(
            self.node_label_event_path, "node_label_event.pkl"
        )
        """
        node_label_event_file_path = os_join(
            self.node_label_event_path, "node_label_event.pkl"
        )
        if os.path.exists(node_label_event_file_path):
            return None

        if not os.path.exists(self.node_label_event_path):
            os.makedirs(self.node_label_event_path)

        # for Karate: fake some
        fk_lb_dim = 10
        total_node_label_events = 100
        edge_list_file_path = os_join(self.edge_event_path, "data.json")
        nodes = []
        lbs = []
        ts = []
        with open(edge_list_file_path, "r") as f:
            for i, line in enumerate(f):
                u, v, w, t = json.loads(line.strip())
                u = str(u)
                v = str(v)
                w = float(w)
                t = float(t)
                nodes.append(u)
                _lb = np.zeros(fk_lb_dim, dtype=bool)
                _lb[np.random.randint(0, fk_lb_dim, 2)] = True
                lbs.append(_lb)
                ts.append(t)
                if len(nodes) == total_node_label_events:
                    break
        fake_lb_events = []

        for i in range(len(nodes)):
            _u = str(nodes[i])
            _lb = lbs[i]
            _t = float(ts[i])
            fake_lb_events.append((_u, _lb, _t))

        with open(node_label_event_file_path, "wb") as f:
            pickle.dump(fake_lb_events, f)

        print(f"write to {node_label_event_file_path}")


class DynGraphReaderPlanetoid(DynGraphReader):
    def __init__(
        self,
        graph_dataset_name: str,
        local_dataset_dir_abs_path: str,
        graph_event_prefix: str = "snapshot-",
        verbose: bool = False,
        use_undirect: bool = False,
    ):
        super().__init__(
            graph_dataset_name=graph_dataset_name,
            local_dataset_dir_abs_path=local_dataset_dir_abs_path,
            graph_event_prefix=graph_event_prefix,
            verbose=verbose,
        )
        self.use_undirect = use_undirect

    def __write_out(self, buffer, f_write):
        f_write.write("\n".join([json.dumps(_) for _ in buffer]))
        f_write.write("\n")

    @override
    def download_raw_data(self, out_buffer_size: int = 10000):
        """download data to local path

        save to self.node_feat_event_path., ...

        """
        # 1: save edge struct
        dataset_pyg = torch_geometric.datasets.Planetoid(
            self.raw_file_path,
            name=self.graph_dataset_name,
            transform=T.NormalizeFeatures(),
        )

        node_set = set()
        node_first_known_seq: List[Tuple[str, float]] = []

        if self.use_undirect:
            edge_set = set()

        raw_data_file_path = os_join(self.raw_file_path, "edge_list.json")
        if not os_exists(raw_data_file_path):
            with open(raw_data_file_path, "w") as f_write:
                out_buffer = []
                _t = 0
                edge_list = dataset_pyg[0].edge_index.numpy().astype(np.int32)
                for i in range(edge_list.shape[1]):
                    u = str(edge_list[0, i])
                    v = str(edge_list[1, i])
                    if self.use_undirect:
                        if (u, v) in edge_set or (v, u) in edge_set:
                            continue
                        else:
                            edge_set.add((u, v))
                    w = float(1.0)
                    t = float(_t)
                    _t += 1

                    # fake node time
                    if u not in node_set:
                        node_set.add(u)
                        node_first_known_seq.append((u, t))
                    if v not in node_set:
                        node_set.add(v)
                        node_first_known_seq.append((v, t))

                    # record/write edge list
                    out_buffer.append((u, v, w, t))

                    if len(out_buffer) >= out_buffer_size:
                        self.__write_out(out_buffer, f_write)
                        out_buffer = []
                if len(out_buffer) > 0:
                    self.__write_out(out_buffer, f_write)
                    out_buffer = []
            print(f"raw edge list created: {raw_data_file_path}")
        else:
            print(f"raw edge list exists: {raw_data_file_path}")

        def __planetoid_node_str_2_id(u: str):
            """dummy function to map node desc to the idx of feature and
            label mat. Planetoid(CORA, CIETEER, PUBMED)'s feat index
            uses node desc (int).

            """

            return int(u)

        # 2: save node feature: pickle of big list of (_u, _feat, _t)
        raw_node_feat_file_path = os_join(
            self.raw_file_path,
            "node_feature_event.pkl",
        )
        if not os_exists(raw_node_feat_file_path):
            # unsorted node feat events
            node_feature_unsorted_events = []
            node_feat_mat = dataset_pyg[0].x.numpy().astype(np.float64)

            if len(node_first_known_seq) != node_feat_mat.shape[0]:
                print(
                    "Warning:"
                    f"nodes in graph {len(node_first_known_seq)}"
                    "!= #node feature"
                    f"({node_feat_mat.shape[0]})."
                    "Remove data folder and try to start from scratch again"
                )

            for u, t in node_first_known_seq:
                # assume: u is idx
                feat = node_feat_mat[__planetoid_node_str_2_id(u), :]
                node_feature_unsorted_events.append((str(u), feat, float(t)))
            with open(raw_node_feat_file_path, "wb") as f_write:
                pickle.dump(node_feature_unsorted_events, f_write)
            print(f"raw node feature created: {raw_node_feat_file_path}")
        else:
            print(f"raw node feature exists: {raw_node_feat_file_path}")

        # 3: save node label: pickle of big list of (_u, _lb, _t)
        raw_node_label_file_path = os_join(
            self.raw_file_path,
            "raw_node_label.pkl",
        )
        if not os_exists(raw_node_label_file_path):
            # unsorted node label events
            node_label_unsorted_events = []
            node_lbs = dataset_pyg[0].y.numpy().astype(np.int16)
            # TODO: output node num_class
            unique_node_class_num = np.unique(node_lbs).shape[0]

            if len(node_first_known_seq) != node_lbs.shape[0]:
                print(
                    "Warning:"
                    f"nodes in graph {len(node_first_known_seq)}"
                    "!= #node labels"
                    f"({node_lbs.shape[0]})."
                    "Remove data folder and try to start from scratch again"
                )

            for u, t in node_first_known_seq:
                lb = node_lbs[__planetoid_node_str_2_id(u)]  #  label as int
                lb_v = np.zeros(unique_node_class_num, dtype=bool)
                lb_v[lb] = True
                node_label_unsorted_events.append((str(u), lb_v, float(t)))
            with open(raw_node_label_file_path, "wb") as f_write:
                pickle.dump(node_label_unsorted_events, f_write)
            print(f"raw node label created: {raw_node_label_file_path}")
        else:
            print(f"raw node label exists: {raw_node_label_file_path}")

    @override
    def parse_edge_event(self, out_buffer_size: int = 10000):
        """parse downloaded raw graph data to edge list, including edge
        timestamp, better to use streaming I/O manner.
        [
            [u: str, v:str, delta_w: float, t:float],
            ...
        ]
        edge_list_file_path = os_join(self.edge_event_path, "data.json")
        """

        raw_data_file_path = os_join(self.raw_file_path, "edge_list.json")
        edge_list_file_path = os_join(self.edge_event_path, "data.json")
        if os.path.exists(edge_list_file_path):
            print(f"unsorted edge list already exists: {edge_list_file_path}")
        else:
            if not os.path.exists(self.edge_event_path):
                if self.verbose:
                    print(
                        f"edge event folder is created: {self.edge_event_path}"
                    )
                os.makedirs(self.edge_event_path)

            out_buffer = []
            with open(edge_list_file_path, "w", encoding="utf-8") as f_write:
                with open(raw_data_file_path, "r", encoding="utf-8") as f_read:
                    for line in f_read:
                        u, v, w, t = json.loads(line.strip())
                        u = str(u)
                        v = str(v)
                        w = float(w)
                        t = float(t)
                        out_buffer.append((u, v, w, t))
                        # NOTE: make it undirected
                        out_buffer.append((v, u, w, t))

                        if len(out_buffer) == out_buffer_size:
                            self.__write_out(out_buffer, f_write)
                            out_buffer = []

                    if len(out_buffer) > 0:
                        self.__write_out(out_buffer, f_write)
                        out_buffer = []
            if self.verbose:
                print(f"edge struct event generated at {edge_list_file_path})")
        return None

    @override
    def parse_node_feature_event(self):
        """convert node feature event (e.g., assign/modify) to list,
        including its timestamp.
        [[ u:str, feature:np.ndarray, t:float ], ...]

        For example: save it to
        node_feat_event_file_path = os_join(
            self.node_feat_event_path, "node_feature_event.pkl"
        )

        """
        node_feat_event_file_path = os_join(
            self.node_feat_event_path, "node_feature_event.pkl"
        )
        if os.path.exists(node_feat_event_file_path):
            print(f"node feature event exists: {node_feat_event_file_path}")
            return None
        if not os.path.exists(self.node_feat_event_path):
            os.makedirs(self.node_feat_event_path)

        raw_node_feat_file_path = os_join(
            self.raw_file_path,
            "node_feature_event.pkl",
        )
        feat_events = []
        with open(raw_node_feat_file_path, "rb") as f:
            raw_node_feature_list = pickle.load(f)

        for i in range(len(raw_node_feature_list)):
            _u = str(raw_node_feature_list[i][0])
            _feat = raw_node_feature_list[i][1].astype(np.float32)
            _t = float(raw_node_feature_list[i][2])
            feat_events.append((_u, _feat, _t))

        with open(node_feat_event_file_path, "wb") as f:
            pickle.dump(feat_events, f)
        print(f"node feature event created: {node_feat_event_file_path}")

    @override
    def parse_node_label_event(self):
        """convert node label event (e.g., assign/modify) to list,
        including its timestamp.
        [[ u:str, label:np.array[Bool], t:float ], ...]

        encode node label into array of int (assuming multi-label
        classification).

        For example: save to
        node_label_event_file_path = os_join(
            self.node_label_event_path, "node_label_event.pkl"
        )
        """
        node_label_event_file_path = os_join(
            self.node_label_event_path, "node_label_event.pkl"
        )
        if os.path.exists(node_label_event_file_path):
            print(f"node label event exists: {node_label_event_file_path}")
            return None

        if not os.path.exists(self.node_label_event_path):
            os.makedirs(self.node_label_event_path)

        raw_node_label_file_path = os_join(
            self.raw_file_path, "raw_node_label.pkl"
        )
        node_label_list = []
        with open(raw_node_label_file_path, "rb") as f:
            raw_node_label_list = pickle.load(f)

        for i in range(len(raw_node_label_list)):
            _u = str(raw_node_label_list[i][0])
            _lb_v: np.ndarray = raw_node_label_list[i][1]
            _t = float(raw_node_label_list[i][2])
            node_label_list.append((_u, _lb_v, _t))

        with open(node_label_event_file_path, "wb") as f:
            pickle.dump(node_label_list, f)
        print(f"node label event created: {node_label_event_file_path}")


class DynGraphReaderEllipticBitcoin(DynGraphReader):
    def __init__(
        self,
        graph_dataset_name: str,
        local_dataset_dir_abs_path: str,
        graph_event_prefix: str = "snapshot-",
        verbose: bool = False,
        already_sort: bool = False,
        use_undirect: bool = False,
    ):
        super().__init__(
            graph_dataset_name=graph_dataset_name,
            local_dataset_dir_abs_path=local_dataset_dir_abs_path,
            graph_event_prefix=graph_event_prefix,
            already_sort=already_sort,
            verbose=verbose,
        )
        self.use_undirect = use_undirect

    def __write_out(self, buffer, f_write):
        f_write.write("\n".join([json.dumps(_) for _ in buffer]))
        f_write.write("\n")

    @override
    def download_raw_data(self, out_buffer_size: int = 10000):
        """download data to local path

        save to self.node_feat_event_path., ...

        """
        # 1: save edge struct
        dataset_pyg = EllipticBitcoinTempDataset(
            self.raw_file_path,
            transform=T.NormalizeFeatures(),
        )

        node_set = set()
        node_first_known_seq: List[Tuple[str, float]] = []

        if self.use_undirect:
            edge_set = set()

        raw_data_file_path = os_join(self.raw_file_path, "edge_list.json")
        if not os_exists(raw_data_file_path):
            with open(raw_data_file_path, "w") as f_write:
                out_buffer = []
                _t = 0
                edge_list = dataset_pyg[0].edge_index.numpy().astype(np.int32)
                for i in range(edge_list.shape[1]):
                    u = str(edge_list[0, i])
                    v = str(edge_list[1, i])
                    if self.use_undirect:
                        if (u, v) in edge_set or (v, u) in edge_set:
                            continue
                        else:
                            edge_set.add((u, v))
                    w = float(1.0)
                    t = float(_t)
                    _t += 1

                    # fake node time
                    if u not in node_set:
                        node_set.add(u)
                        node_first_known_seq.append((u, t))
                    if v not in node_set:
                        node_set.add(v)
                        node_first_known_seq.append((v, t))

                    # record/write edge list
                    out_buffer.append((u, v, w, t))

                    if len(out_buffer) >= out_buffer_size:
                        self.__write_out(out_buffer, f_write)
                        out_buffer = []
                if len(out_buffer) > 0:
                    self.__write_out(out_buffer, f_write)
                    out_buffer = []
            print(f"raw edge list created: {raw_data_file_path}")
        else:
            print(f"raw edge list exists: {raw_data_file_path}")

        def __planetoid_node_str_2_id(u: str):
            """dummy function to map node desc to the idx of feature and
            label mat. Planetoid(CORA, CIETEER, PUBMED)'s feat index
            uses node desc (int).

            """

            return int(u)

        # 2: save node feature: pickle of big list of (_u, _feat, _t)
        raw_node_feat_file_path = os_join(
            self.raw_file_path,
            "node_feature_event.pkl",
        )
        if not os_exists(raw_node_feat_file_path):
            # unsorted node feat events
            node_feature_unsorted_events = []
            node_feat_mat = dataset_pyg[0].x.numpy().astype(np.float64)

            if len(node_first_known_seq) != node_feat_mat.shape[0]:
                print(
                    "Warning:"
                    f"nodes in graph {len(node_first_known_seq)}"
                    "!= #node feature"
                    f"({node_feat_mat.shape[0]})."
                    "Remove data folder and try to start from scratch again"
                )

            for u, t in node_first_known_seq:
                # assume: u is idx
                feat = node_feat_mat[__planetoid_node_str_2_id(u), :]
                node_feature_unsorted_events.append((str(u), feat, float(t)))
            with open(raw_node_feat_file_path, "wb") as f_write:
                pickle.dump(node_feature_unsorted_events, f_write)
            print(f"raw node feature created: {raw_node_feat_file_path}")
        else:
            print(f"raw node feature exists: {raw_node_feat_file_path}")

        # 3: save node label: pickle of big list of (_u, _lb, _t)
        raw_node_label_file_path = os_join(
            self.raw_file_path,
            "raw_node_label.pkl",
        )
        if not os_exists(raw_node_label_file_path):
            # unsorted node label events
            node_label_unsorted_events = []
            node_lbs = dataset_pyg[0].y.numpy().astype(np.int16)
            # NOTE: in bitcoin dataset: remove 2 ("unknown")
            unique_node_class_num = np.unique(node_lbs).shape[0] - 1

            if len(node_first_known_seq) != node_lbs.shape[0]:
                print(
                    "Warning:"
                    f"nodes in graph {len(node_first_known_seq)}"
                    "!= #node labels"
                    f"({node_lbs.shape[0]})."
                    "Remove data folder and try to start from scratch again"
                )

            for u, t in node_first_known_seq:
                lb = node_lbs[__planetoid_node_str_2_id(u)]  #  label as int
                lb_v = np.zeros(unique_node_class_num, dtype=bool)
                if lb != 2:  # NOTE: skip 2 --> unknown
                    lb_v[lb] = True
                node_label_unsorted_events.append((str(u), lb_v, float(t)))
            with open(raw_node_label_file_path, "wb") as f_write:
                pickle.dump(node_label_unsorted_events, f_write)
            print(f"raw node label created: {raw_node_label_file_path}")
        else:
            print(f"raw node label exists: {raw_node_label_file_path}")

    @override
    def parse_edge_event(self, out_buffer_size: int = 10000):
        """parse downloaded raw graph data to edge list, including edge
        timestamp, better to use streaming I/O manner.
        [
            [u: str, v:str, delta_w: float, t:float],
            ...
        ]
        edge_list_file_path = os_join(self.edge_event_path, "data.json")
        """

        raw_data_file_path = os_join(self.raw_file_path, "edge_list.json")
        edge_list_file_path = os_join(self.edge_event_path, "data.json")
        if os.path.exists(edge_list_file_path):
            print(f"unsorted edge list already exists: {edge_list_file_path}")
        else:
            if not os.path.exists(self.edge_event_path):
                if self.verbose:
                    print(
                        f"edge event folder is created: {self.edge_event_path}"
                    )
                os.makedirs(self.edge_event_path)

            out_buffer = []
            with open(edge_list_file_path, "w", encoding="utf-8") as f_write:
                with open(raw_data_file_path, "r", encoding="utf-8") as f_read:
                    for line in f_read:
                        u, v, w, t = json.loads(line.strip())
                        u = str(u)
                        v = str(v)
                        w = float(w)
                        t = float(t)
                        out_buffer.append((u, v, w, t))
                        # NOTE: make it undirected
                        out_buffer.append((v, u, w, t))

                        if len(out_buffer) == out_buffer_size:
                            self.__write_out(out_buffer, f_write)
                            out_buffer = []

                    if len(out_buffer) > 0:
                        self.__write_out(out_buffer, f_write)
                        out_buffer = []
            if self.verbose:
                print(f"edge struct event generated at {edge_list_file_path})")
        return None

    @override
    def parse_node_feature_event(self):
        """convert node feature event (e.g., assign/modify) to list,
        including its timestamp.
        [[ u:str, feature:np.ndarray, t:float ], ...]

        For example: save it to
        node_feat_event_file_path = os_join(
            self.node_feat_event_path, "node_feature_event.pkl"
        )

        """
        node_feat_event_file_path = os_join(
            self.node_feat_event_path, "node_feature_event.pkl"
        )
        if os.path.exists(node_feat_event_file_path):
            print(f"node feature event exists: {node_feat_event_file_path}")
            return None
        if not os.path.exists(self.node_feat_event_path):
            os.makedirs(self.node_feat_event_path)

        raw_node_feat_file_path = os_join(
            self.raw_file_path,
            "node_feature_event.pkl",
        )
        feat_events = []
        with open(raw_node_feat_file_path, "rb") as f:
            raw_node_feature_list = pickle.load(f)

        for i in range(len(raw_node_feature_list)):
            _u = str(raw_node_feature_list[i][0])
            _feat = raw_node_feature_list[i][1].astype(np.float32)
            _t = float(raw_node_feature_list[i][2])
            feat_events.append((_u, _feat, _t))

        with open(node_feat_event_file_path, "wb") as f:
            pickle.dump(feat_events, f)
        print(f"node feature event created: {node_feat_event_file_path}")

    @override
    def parse_node_label_event(self):
        """convert node label event (e.g., assign/modify) to list,
        including its timestamp.
        [[ u:str, label:np.array[Bool], t:float ], ...]

        encode node label into array of int (assuming multi-label
        classification).

        For example: save to
        node_label_event_file_path = os_join(
            self.node_label_event_path, "node_label_event.pkl"
        )
        """
        node_label_event_file_path = os_join(
            self.node_label_event_path, "node_label_event.pkl"
        )
        if os.path.exists(node_label_event_file_path):
            print(f"node label event exists: {node_label_event_file_path}")
            return None

        if not os.path.exists(self.node_label_event_path):
            os.makedirs(self.node_label_event_path)

        raw_node_label_file_path = os_join(
            self.raw_file_path, "raw_node_label.pkl"
        )
        node_label_list = []
        with open(raw_node_label_file_path, "rb") as f:
            raw_node_label_list = pickle.load(f)

        for i in range(len(raw_node_label_list)):
            _u = str(raw_node_label_list[i][0])
            _lb_v: np.ndarray = raw_node_label_list[i][1]
            _t = float(raw_node_label_list[i][2])
            node_label_list.append((_u, _lb_v, _t))

        with open(node_label_event_file_path, "wb") as f:
            pickle.dump(node_label_list, f)
        print(f"node label event created: {node_label_event_file_path}")


class DynGraphReaderAttributedGraph(DynGraphReader):
    def __init__(
        self,
        graph_dataset_name: str,  # BlogCatalog
        local_dataset_dir_abs_path: str,
        graph_event_prefix: str = "snapshot-",
        verbose: bool = False,
        use_undirect: bool = False,
    ):
        super().__init__(
            graph_dataset_name=graph_dataset_name,
            local_dataset_dir_abs_path=local_dataset_dir_abs_path,
            graph_event_prefix=graph_event_prefix,
            verbose=verbose,
        )
        self.use_undirect = use_undirect

    def __write_out(self, buffer, f_write):
        f_write.write("\n".join([json.dumps(_) for _ in buffer]))
        f_write.write("\n")

    @override
    def download_raw_data(self, out_buffer_size: int = 10000):
        """download data to local path

        save to self.node_feat_event_path., ...

        """
        # 1: save edge struct
        dataset_pyg = torch_geometric.datasets.AttributedGraphDataset(
            root=self.raw_file_path,
            name=self.graph_dataset_name,
            transform=T.NormalizeFeatures(),
        )

        node_set = set()
        node_first_known_seq: List[Tuple[str, float]] = []
        if self.use_undirect:
            edge_set = set()

        raw_data_file_path = os_join(self.raw_file_path, "edge_list.json")
        if not os_exists(raw_data_file_path):
            with open(raw_data_file_path, "w") as f_write:
                out_buffer = []
                edge_list = dataset_pyg[0].edge_index.numpy().astype(np.int32)
                _t = 0
                for i in range(edge_list.shape[1]):
                    u = str(edge_list[0, i])
                    v = str(edge_list[1, i])
                    if self.use_undirect:
                        if (u, v) in edge_set or (v, u) in edge_set:
                            continue
                        else:
                            edge_set.add((u, v))
                    w = float(1.0)
                    t = float(_t)
                    _t += 1

                    # fake node time
                    if u not in node_set:
                        node_set.add(u)
                        node_first_known_seq.append((u, t))
                    if v not in node_set:
                        node_set.add(v)
                        node_first_known_seq.append((v, t))

                    # record/write edge list
                    out_buffer.append((u, v, w, t))

                    if len(out_buffer) >= out_buffer_size:
                        self.__write_out(out_buffer, f_write)
                        out_buffer = []
                if len(out_buffer) > 0:
                    self.__write_out(out_buffer, f_write)
                    out_buffer = []
            print(f"raw edge list created: {raw_data_file_path}")
        else:
            print(f"raw edge list exists: {raw_data_file_path}")

        def __planetoid_node_str_2_id(u: str):
            """dummy function to map node desc to the idx of feature and
            label mat. Planetoid(CORA, CIETEER, PUBMED)'s feat index
            uses node desc (int).

            """

            return int(u)

        # 2: save node feature: pickle of big list of (_u, _feat, _t)
        raw_node_feat_file_path = os_join(
            self.raw_file_path,
            "node_feature_event.pkl",
        )
        if not os_exists(raw_node_feat_file_path):
            # unsorted node feat events
            node_feature_unsorted_events = []
            node_feat_mat = dataset_pyg[0].x.numpy().astype(np.float64)

            if len(node_first_known_seq) != node_feat_mat.shape[0]:
                print(
                    "Warning:"
                    f"nodes in graph {len(node_first_known_seq)}"
                    "!= #node feature"
                    f"({node_feat_mat.shape[0]})."
                    "Remove data folder and try to start from scratch again"
                )

            for u, t in node_first_known_seq:
                # assume: u is idx
                feat = node_feat_mat[__planetoid_node_str_2_id(u), :]
                node_feature_unsorted_events.append((str(u), feat, float(t)))
            with open(raw_node_feat_file_path, "wb") as f_write:
                pickle.dump(node_feature_unsorted_events, f_write)
            print(f"raw node feature created: {raw_node_feat_file_path}")
        else:
            print(f"raw node feature exists: {raw_node_feat_file_path}")

        # 3: save node label: pickle of big list of (_u, _lb, _t)
        raw_node_label_file_path = os_join(
            self.raw_file_path,
            "raw_node_label.pkl",
        )
        if not os_exists(raw_node_label_file_path):
            # unsorted node label events
            node_label_unsorted_events = []
            node_lbs = dataset_pyg[0].y.numpy().astype(np.int16)
            # TODO: output node num_class
            unique_node_class_num = np.unique(node_lbs).shape[0]

            if len(node_first_known_seq) != node_lbs.shape[0]:
                print(
                    "Warning:"
                    f"nodes in graph {len(node_first_known_seq)}"
                    "!= #node labels"
                    f"({node_lbs.shape[0]})."
                    "Remove data folder and try to start from scratch again"
                )

            for u, t in node_first_known_seq:
                lb = node_lbs[__planetoid_node_str_2_id(u)]  #  label as int
                lb_v = np.zeros(unique_node_class_num, dtype=bool)
                lb_v[lb] = True
                node_label_unsorted_events.append((str(u), lb_v, float(t)))
            with open(raw_node_label_file_path, "wb") as f_write:
                pickle.dump(node_label_unsorted_events, f_write)
            print(f"raw node label created: {raw_node_label_file_path}")
        else:
            print(f"raw node label exists: {raw_node_label_file_path}")

    @override
    def parse_edge_event(self, out_buffer_size: int = 10000):
        """parse downloaded raw graph data to edge list, including edge
        timestamp, better to use streaming I/O manner.
        [
            [u: str, v:str, delta_w: float, t:float],
            ...
        ]
        edge_list_file_path = os_join(self.edge_event_path, "data.json")
        """

        raw_data_file_path = os_join(self.raw_file_path, "edge_list.json")
        edge_list_file_path = os_join(self.edge_event_path, "data.json")
        if os.path.exists(edge_list_file_path):
            print(f"unsorted edge list already exists: {edge_list_file_path}")
        else:
            if not os.path.exists(self.edge_event_path):
                if self.verbose:
                    print(
                        f"edge event folder is created: {self.edge_event_path}"
                    )
                os.makedirs(self.edge_event_path)

            out_buffer = []
            with open(edge_list_file_path, "w", encoding="utf-8") as f_write:
                with open(raw_data_file_path, "r", encoding="utf-8") as f_read:
                    for line in f_read:
                        u, v, w, t = json.loads(line.strip())
                        u = str(u)
                        v = str(v)
                        w = float(w)
                        t = float(t)
                        out_buffer.append((u, v, w, t))
                        # NOTE: make it undirected
                        out_buffer.append((v, u, w, t))

                        if len(out_buffer) == out_buffer_size:
                            self.__write_out(out_buffer, f_write)
                            out_buffer = []

                    if len(out_buffer) > 0:
                        self.__write_out(out_buffer, f_write)
                        out_buffer = []
            if self.verbose:
                print(f"edge struct event generated at {edge_list_file_path})")
        return None

    @override
    def parse_node_feature_event(self):
        """convert node feature event (e.g., assign/modify) to list,
        including its timestamp.
        [[ u:str, feature:np.ndarray, t:float ], ...]

        For example: save it to
        node_feat_event_file_path = os_join(
            self.node_feat_event_path, "node_feature_event.pkl"
        )

        """
        node_feat_event_file_path = os_join(
            self.node_feat_event_path, "node_feature_event.pkl"
        )
        if os.path.exists(node_feat_event_file_path):
            print(f"node feature event exists: {node_feat_event_file_path}")
            return None
        if not os.path.exists(self.node_feat_event_path):
            os.makedirs(self.node_feat_event_path)

        raw_node_feat_file_path = os_join(
            self.raw_file_path,
            "node_feature_event.pkl",
        )
        feat_events = []
        with open(raw_node_feat_file_path, "rb") as f:
            raw_node_feature_list = pickle.load(f)

        for i in range(len(raw_node_feature_list)):
            _u = str(raw_node_feature_list[i][0])
            _feat = raw_node_feature_list[i][1].astype(np.float32)
            _t = float(raw_node_feature_list[i][2])
            feat_events.append((_u, _feat, _t))

        with open(node_feat_event_file_path, "wb") as f:
            pickle.dump(feat_events, f)
        print(f"node feature event created: {node_feat_event_file_path}")

    @override
    def parse_node_label_event(self):
        """convert node label event (e.g., assign/modify) to list,
        including its timestamp.
        [[ u:str, label:np.array[Bool], t:float ], ...]

        encode node label into array of int (assuming multi-label
        classification).

        For example: save to
        node_label_event_file_path = os_join(
            self.node_label_event_path, "node_label_event.pkl"
        )
        """
        node_label_event_file_path = os_join(
            self.node_label_event_path, "node_label_event.pkl"
        )
        if os.path.exists(node_label_event_file_path):
            print(f"node label event exists: {node_label_event_file_path}")
            return None

        if not os.path.exists(self.node_label_event_path):
            os.makedirs(self.node_label_event_path)

        raw_node_label_file_path = os_join(
            self.raw_file_path, "raw_node_label.pkl"
        )
        node_label_list = []
        with open(raw_node_label_file_path, "rb") as f:
            raw_node_label_list = pickle.load(f)

        for i in range(len(raw_node_label_list)):
            _u = str(raw_node_label_list[i][0])
            _lb_v: np.ndarray = raw_node_label_list[i][1]
            _t = float(raw_node_label_list[i][2])
            node_label_list.append((_u, _lb_v, _t))

        with open(node_label_event_file_path, "wb") as f:
            pickle.dump(node_label_list, f)
        print(f"node label event created: {node_label_event_file_path}")


class DynGraphReaderWikiCS(DynGraphReader):
    def __init__(
        self,
        graph_dataset_name: str,  # DBLP
        local_dataset_dir_abs_path: str,
        graph_event_prefix: str = "snapshot-",
        verbose: bool = False,
        use_undirect: bool = False,
    ):
        super().__init__(
            graph_dataset_name=graph_dataset_name,
            local_dataset_dir_abs_path=local_dataset_dir_abs_path,
            graph_event_prefix=graph_event_prefix,
            verbose=verbose,
        )
        self.use_undirect = use_undirect

    def __write_out(self, buffer, f_write):
        f_write.write("\n".join([json.dumps(_) for _ in buffer]))
        f_write.write("\n")

    @override
    def download_raw_data(self, out_buffer_size: int = 10000):
        """download data to local path

        save to self.node_feat_event_path., ...

        """
        # 1: save edge struct
        dataset_pyg = torch_geometric.datasets.WikiCS(
            root=self.raw_file_path,
            transform=T.NormalizeFeatures(),
        )

        node_set = set()
        node_first_known_seq: List[Tuple[str, float]] = []
        if self.use_undirect:
            edge_set = set()

        raw_data_file_path = os_join(self.raw_file_path, "edge_list.json")
        if not os_exists(raw_data_file_path):
            with open(raw_data_file_path, "w") as f_write:
                out_buffer = []
                edge_list = dataset_pyg[0].edge_index.numpy().astype(np.int32)
                _t = 0
                for i in range(edge_list.shape[1]):
                    u = str(edge_list[0, i])
                    v = str(edge_list[1, i])
                    if self.use_undirect:
                        if (u, v) in edge_set or (v, u) in edge_set:
                            continue
                        else:
                            edge_set.add((u, v))
                    w = float(1.0)
                    t = float(_t)
                    _t += 1

                    # fake node time
                    if u not in node_set:
                        node_set.add(u)
                        node_first_known_seq.append((u, t))
                    if v not in node_set:
                        node_set.add(v)
                        node_first_known_seq.append((v, t))

                    # record/write edge list
                    out_buffer.append((u, v, w, t))

                    if len(out_buffer) >= out_buffer_size:
                        self.__write_out(out_buffer, f_write)
                        out_buffer = []
                if len(out_buffer) > 0:
                    self.__write_out(out_buffer, f_write)
                    out_buffer = []
            print(f"raw edge list created: {raw_data_file_path}")
        else:
            print(f"raw edge list exists: {raw_data_file_path}")

        def __planetoid_node_str_2_id(u: str):
            """dummy function to map node desc to the idx of feature and
            label mat. Planetoid(CORA, CIETEER, PUBMED)'s feat index
            uses node desc (int).

            """

            return int(u)

        # 2: save node feature: pickle of big list of (_u, _feat, _t)
        raw_node_feat_file_path = os_join(
            self.raw_file_path,
            "node_feature_event.pkl",
        )
        if not os_exists(raw_node_feat_file_path):
            # unsorted node feat events
            node_feature_unsorted_events = []
            node_feat_mat = dataset_pyg[0].x.numpy().astype(np.float64)

            if len(node_first_known_seq) != node_feat_mat.shape[0]:
                print(
                    "Warning:"
                    f"nodes in graph {len(node_first_known_seq)}"
                    "!= #node feature"
                    f"({node_feat_mat.shape[0]})."
                    "Remove data folder and try to start from scratch again"
                )

            for u, t in node_first_known_seq:
                # assume: u is idx
                feat = node_feat_mat[__planetoid_node_str_2_id(u), :]
                node_feature_unsorted_events.append((str(u), feat, float(t)))
            with open(raw_node_feat_file_path, "wb") as f_write:
                pickle.dump(node_feature_unsorted_events, f_write)
            print(f"raw node feature created: {raw_node_feat_file_path}")
        else:
            print(f"raw node feature exists: {raw_node_feat_file_path}")

        # 3: save node label: pickle of big list of (_u, _lb, _t)
        raw_node_label_file_path = os_join(
            self.raw_file_path,
            "raw_node_label.pkl",
        )
        if not os_exists(raw_node_label_file_path):
            # unsorted node label events
            node_label_unsorted_events = []
            node_lbs = dataset_pyg[0].y.numpy().astype(np.int16)
            # TODO: output node num_class
            unique_node_class_num = np.unique(node_lbs).shape[0]

            if len(node_first_known_seq) != node_lbs.shape[0]:
                print(
                    "Warning:"
                    f"nodes in graph {len(node_first_known_seq)}"
                    "!= #node labels"
                    f"({node_lbs.shape[0]})."
                    "Remove data folder and try to start from scratch again"
                )

            for u, t in node_first_known_seq:
                lb = node_lbs[__planetoid_node_str_2_id(u)]  #  label as int
                lb_v = np.zeros(unique_node_class_num, dtype=bool)
                lb_v[lb] = True
                node_label_unsorted_events.append((str(u), lb_v, float(t)))
            with open(raw_node_label_file_path, "wb") as f_write:
                pickle.dump(node_label_unsorted_events, f_write)
            print(f"raw node label created: {raw_node_label_file_path}")
        else:
            print(f"raw node label exists: {raw_node_label_file_path}")

    @override
    def parse_edge_event(self, out_buffer_size: int = 10000):
        """parse downloaded raw graph data to edge list, including edge
        timestamp, better to use streaming I/O manner.
        [
            [u: str, v:str, delta_w: float, t:float],
            ...
        ]
        edge_list_file_path = os_join(self.edge_event_path, "data.json")
        """

        raw_data_file_path = os_join(self.raw_file_path, "edge_list.json")
        edge_list_file_path = os_join(self.edge_event_path, "data.json")
        if os.path.exists(edge_list_file_path):
            print(f"unsorted edge list already exists: {edge_list_file_path}")
        else:
            if not os.path.exists(self.edge_event_path):
                if self.verbose:
                    print(
                        f"edge event folder is created: {self.edge_event_path}"
                    )
                os.makedirs(self.edge_event_path)

            out_buffer = []
            with open(edge_list_file_path, "w", encoding="utf-8") as f_write:
                with open(raw_data_file_path, "r", encoding="utf-8") as f_read:
                    for line in f_read:
                        u, v, w, t = json.loads(line.strip())
                        u = str(u)
                        v = str(v)
                        w = float(w)
                        t = float(t)
                        out_buffer.append((u, v, w, t))
                        # NOTE: make it undirected
                        out_buffer.append((v, u, w, t))

                        if len(out_buffer) == out_buffer_size:
                            self.__write_out(out_buffer, f_write)
                            out_buffer = []

                    if len(out_buffer) > 0:
                        self.__write_out(out_buffer, f_write)
                        out_buffer = []
            if self.verbose:
                print(f"edge struct event generated at {edge_list_file_path})")
        return None

    @override
    def parse_node_feature_event(self):
        """convert node feature event (e.g., assign/modify) to list,
        including its timestamp.
        [[ u:str, feature:np.ndarray, t:float ], ...]

        For example: save it to
        node_feat_event_file_path = os_join(
            self.node_feat_event_path, "node_feature_event.pkl"
        )

        """
        node_feat_event_file_path = os_join(
            self.node_feat_event_path, "node_feature_event.pkl"
        )
        if os.path.exists(node_feat_event_file_path):
            print(f"node feature event exists: {node_feat_event_file_path}")
            return None
        if not os.path.exists(self.node_feat_event_path):
            os.makedirs(self.node_feat_event_path)

        raw_node_feat_file_path = os_join(
            self.raw_file_path,
            "node_feature_event.pkl",
        )
        feat_events = []
        with open(raw_node_feat_file_path, "rb") as f:
            raw_node_feature_list = pickle.load(f)

        for i in range(len(raw_node_feature_list)):
            _u = str(raw_node_feature_list[i][0])
            _feat = raw_node_feature_list[i][1].astype(np.float32)
            _t = float(raw_node_feature_list[i][2])
            feat_events.append((_u, _feat, _t))

        with open(node_feat_event_file_path, "wb") as f:
            pickle.dump(feat_events, f)
        print(f"node feature event created: {node_feat_event_file_path}")

    @override
    def parse_node_label_event(self):
        """convert node label event (e.g., assign/modify) to list,
        including its timestamp.
        [[ u:str, label:np.array[Bool], t:float ], ...]

        encode node label into array of int (assuming multi-label
        classification).

        For example: save to
        node_label_event_file_path = os_join(
            self.node_label_event_path, "node_label_event.pkl"
        )
        """
        node_label_event_file_path = os_join(
            self.node_label_event_path, "node_label_event.pkl"
        )
        if os.path.exists(node_label_event_file_path):
            print(f"node label event exists: {node_label_event_file_path}")
            return None

        if not os.path.exists(self.node_label_event_path):
            os.makedirs(self.node_label_event_path)

        raw_node_label_file_path = os_join(
            self.raw_file_path, "raw_node_label.pkl"
        )
        node_label_list = []
        with open(raw_node_label_file_path, "rb") as f:
            raw_node_label_list = pickle.load(f)

        for i in range(len(raw_node_label_list)):
            _u = str(raw_node_label_list[i][0])
            _lb_v: np.ndarray = raw_node_label_list[i][1]
            _t = float(raw_node_label_list[i][2])
            node_label_list.append((_u, _lb_v, _t))

        with open(node_label_event_file_path, "wb") as f:
            pickle.dump(node_label_list, f)
        print(f"node label event created: {node_label_event_file_path}")


class DynGraphReaderCitationFull(DynGraphReader):
    def __init__(
        self,
        graph_dataset_name: str,  # DBLP
        local_dataset_dir_abs_path: str,
        graph_event_prefix: str = "snapshot-",
        verbose: bool = False,
        use_undirect: bool = False,
    ):
        super().__init__(
            graph_dataset_name=graph_dataset_name,
            local_dataset_dir_abs_path=local_dataset_dir_abs_path,
            graph_event_prefix=graph_event_prefix,
            verbose=verbose,
        )
        self.use_undirect = use_undirect

    def __write_out(self, buffer, f_write):
        f_write.write("\n".join([json.dumps(_) for _ in buffer]))
        f_write.write("\n")

    @override
    def download_raw_data(self, out_buffer_size: int = 10000):
        """download data to local path

        save to self.node_feat_event_path., ...

        """
        # 1: save edge struct
        dataset_pyg = torch_geometric.datasets.CitationFull(
            root=self.raw_file_path,
            name = self.graph_dataset_name,
            transform=T.NormalizeFeatures(),
        )

        node_set = set()
        node_first_known_seq: List[Tuple[str, float]] = []
        if self.use_undirect:
            edge_set = set()

        raw_data_file_path = os_join(self.raw_file_path, "edge_list.json")
        if not os_exists(raw_data_file_path):
            with open(raw_data_file_path, "w") as f_write:
                out_buffer = []
                edge_list = dataset_pyg[0].edge_index.numpy().astype(np.int32)
                _t = 0
                for i in range(edge_list.shape[1]):
                    u = str(edge_list[0, i])
                    v = str(edge_list[1, i])
                    if self.use_undirect:
                        if (u, v) in edge_set or (v, u) in edge_set:
                            continue
                        else:
                            edge_set.add((u, v))
                    w = float(1.0)
                    t = float(_t)
                    _t += 1

                    # fake node time
                    if u not in node_set:
                        node_set.add(u)
                        node_first_known_seq.append((u, t))
                    if v not in node_set:
                        node_set.add(v)
                        node_first_known_seq.append((v, t))

                    # record/write edge list
                    out_buffer.append((u, v, w, t))

                    if len(out_buffer) >= out_buffer_size:
                        self.__write_out(out_buffer, f_write)
                        out_buffer = []
                if len(out_buffer) > 0:
                    self.__write_out(out_buffer, f_write)
                    out_buffer = []
            print(f"raw edge list created: {raw_data_file_path}")
        else:
            print(f"raw edge list exists: {raw_data_file_path}")

        def __planetoid_node_str_2_id(u: str):
            """dummy function to map node desc to the idx of feature and
            label mat. Planetoid(CORA, CIETEER, PUBMED)'s feat index
            uses node desc (int).

            """

            return int(u)

        # 2: save node feature: pickle of big list of (_u, _feat, _t)
        raw_node_feat_file_path = os_join(
            self.raw_file_path,
            "node_feature_event.pkl",
        )
        if not os_exists(raw_node_feat_file_path):
            # unsorted node feat events
            node_feature_unsorted_events = []
            node_feat_mat = dataset_pyg[0].x.numpy().astype(np.float64)

            if len(node_first_known_seq) != node_feat_mat.shape[0]:
                print(
                    "Warning:"
                    f"nodes in graph {len(node_first_known_seq)}"
                    "!= #node feature"
                    f"({node_feat_mat.shape[0]})."
                    "Remove data folder and try to start from scratch again"
                )

            for u, t in node_first_known_seq:
                # assume: u is idx
                feat = node_feat_mat[__planetoid_node_str_2_id(u), :]
                node_feature_unsorted_events.append((str(u), feat, float(t)))
            with open(raw_node_feat_file_path, "wb") as f_write:
                pickle.dump(node_feature_unsorted_events, f_write)
            print(f"raw node feature created: {raw_node_feat_file_path}")
        else:
            print(f"raw node feature exists: {raw_node_feat_file_path}")

        # 3: save node label: pickle of big list of (_u, _lb, _t)
        raw_node_label_file_path = os_join(
            self.raw_file_path,
            "raw_node_label.pkl",
        )
        if not os_exists(raw_node_label_file_path):
            # unsorted node label events
            node_label_unsorted_events = []
            node_lbs = dataset_pyg[0].y.numpy().astype(np.int16)
            # TODO: output node num_class
            unique_node_class_num = np.unique(node_lbs).shape[0]

            if len(node_first_known_seq) != node_lbs.shape[0]:
                print(
                    "Warning:"
                    f"nodes in graph {len(node_first_known_seq)}"
                    "!= #node labels"
                    f"({node_lbs.shape[0]})."
                    "Remove data folder and try to start from scratch again"
                )

            for u, t in node_first_known_seq:
                lb = node_lbs[__planetoid_node_str_2_id(u)]  #  label as int
                lb_v = np.zeros(unique_node_class_num, dtype=bool)
                lb_v[lb] = True
                node_label_unsorted_events.append((str(u), lb_v, float(t)))
            with open(raw_node_label_file_path, "wb") as f_write:
                pickle.dump(node_label_unsorted_events, f_write)
            print(f"raw node label created: {raw_node_label_file_path}")
        else:
            print(f"raw node label exists: {raw_node_label_file_path}")

    @override
    def parse_edge_event(self, out_buffer_size: int = 10000):
        """parse downloaded raw graph data to edge list, including edge
        timestamp, better to use streaming I/O manner.
        [
            [u: str, v:str, delta_w: float, t:float],
            ...
        ]
        edge_list_file_path = os_join(self.edge_event_path, "data.json")
        """

        raw_data_file_path = os_join(self.raw_file_path, "edge_list.json")
        edge_list_file_path = os_join(self.edge_event_path, "data.json")
        if os.path.exists(edge_list_file_path):
            print(f"unsorted edge list already exists: {edge_list_file_path}")
        else:
            if not os.path.exists(self.edge_event_path):
                if self.verbose:
                    print(
                        f"edge event folder is created: {self.edge_event_path}"
                    )
                os.makedirs(self.edge_event_path)

            out_buffer = []
            with open(edge_list_file_path, "w", encoding="utf-8") as f_write:
                with open(raw_data_file_path, "r", encoding="utf-8") as f_read:
                    for line in f_read:
                        u, v, w, t = json.loads(line.strip())
                        u = str(u)
                        v = str(v)
                        w = float(w)
                        t = float(t)
                        out_buffer.append((u, v, w, t))
                        # NOTE: make it undirected
                        out_buffer.append((v, u, w, t))

                        if len(out_buffer) == out_buffer_size:
                            self.__write_out(out_buffer, f_write)
                            out_buffer = []

                    if len(out_buffer) > 0:
                        self.__write_out(out_buffer, f_write)
                        out_buffer = []
            if self.verbose:
                print(f"edge struct event generated at {edge_list_file_path})")
        return None

    @override
    def parse_node_feature_event(self):
        """convert node feature event (e.g., assign/modify) to list,
        including its timestamp.
        [[ u:str, feature:np.ndarray, t:float ], ...]

        For example: save it to
        node_feat_event_file_path = os_join(
            self.node_feat_event_path, "node_feature_event.pkl"
        )

        """
        node_feat_event_file_path = os_join(
            self.node_feat_event_path, "node_feature_event.pkl"
        )
        if os.path.exists(node_feat_event_file_path):
            print(f"node feature event exists: {node_feat_event_file_path}")
            return None
        if not os.path.exists(self.node_feat_event_path):
            os.makedirs(self.node_feat_event_path)

        raw_node_feat_file_path = os_join(
            self.raw_file_path,
            "node_feature_event.pkl",
        )
        feat_events = []
        with open(raw_node_feat_file_path, "rb") as f:
            raw_node_feature_list = pickle.load(f)

        for i in range(len(raw_node_feature_list)):
            _u = str(raw_node_feature_list[i][0])
            _feat = raw_node_feature_list[i][1].astype(np.float32)
            _t = float(raw_node_feature_list[i][2])
            feat_events.append((_u, _feat, _t))

        with open(node_feat_event_file_path, "wb") as f:
            pickle.dump(feat_events, f)
        print(f"node feature event created: {node_feat_event_file_path}")

    @override
    def parse_node_label_event(self):
        """convert node label event (e.g., assign/modify) to list,
        including its timestamp.
        [[ u:str, label:np.array[Bool], t:float ], ...]

        encode node label into array of int (assuming multi-label
        classification).

        For example: save to
        node_label_event_file_path = os_join(
            self.node_label_event_path, "node_label_event.pkl"
        )
        """
        node_label_event_file_path = os_join(
            self.node_label_event_path, "node_label_event.pkl"
        )
        if os.path.exists(node_label_event_file_path):
            print(f"node label event exists: {node_label_event_file_path}")
            return None

        if not os.path.exists(self.node_label_event_path):
            os.makedirs(self.node_label_event_path)

        raw_node_label_file_path = os_join(
            self.raw_file_path, "raw_node_label.pkl"
        )
        node_label_list = []
        with open(raw_node_label_file_path, "rb") as f:
            raw_node_label_list = pickle.load(f)

        for i in range(len(raw_node_label_list)):
            _u = str(raw_node_label_list[i][0])
            _lb_v: np.ndarray = raw_node_label_list[i][1]
            _t = float(raw_node_label_list[i][2])
            node_label_list.append((_u, _lb_v, _t))

        with open(node_label_event_file_path, "wb") as f:
            pickle.dump(node_label_list, f)
        print(f"node label event created: {node_label_event_file_path}")


class DynGraphReaderCoauthor(DynGraphReader):
    def __init__(
        self,
        graph_dataset_name: str,  # Physics CS
        local_dataset_dir_abs_path: str,
        graph_event_prefix: str = "snapshot-",
        verbose: bool = False,
        use_undirect: bool = False,
    ):
        super().__init__(
            graph_dataset_name=graph_dataset_name,
            local_dataset_dir_abs_path=local_dataset_dir_abs_path,
            graph_event_prefix=graph_event_prefix,
            verbose=verbose,
        )
        self.use_undirect = use_undirect

    def __write_out(self, buffer, f_write):
        f_write.write("\n".join([json.dumps(_) for _ in buffer]))
        f_write.write("\n")

    @override
    def download_raw_data(self, out_buffer_size: int = 10000):
        """download data to local path

        save to self.node_feat_event_path., ...

        """
        # 1: save edge struct
        dataset_pyg = torch_geometric.datasets.Coauthor(
            root=self.raw_file_path,
            name=self.graph_dataset_name,
            transform=T.NormalizeFeatures(),
        )

        node_set = set()
        node_first_known_seq: List[Tuple[str, float]] = []
        if self.use_undirect:
            edge_set = set()

        raw_data_file_path = os_join(self.raw_file_path, "edge_list.json")
        if not os_exists(raw_data_file_path):
            with open(raw_data_file_path, "w") as f_write:
                out_buffer = []
                edge_list = dataset_pyg[0].edge_index.numpy().astype(np.int32)
                _t = 0
                for i in range(edge_list.shape[1]):
                    u = str(edge_list[0, i])
                    v = str(edge_list[1, i])
                    if self.use_undirect:
                        if (u, v) in edge_set or (v, u) in edge_set:
                            continue
                        else:
                            edge_set.add((u, v))
                    w = float(1.0)
                    t = float(_t)
                    _t += 1

                    # fake node time
                    if u not in node_set:
                        node_set.add(u)
                        node_first_known_seq.append((u, t))
                    if v not in node_set:
                        node_set.add(v)
                        node_first_known_seq.append((v, t))

                    # record/write edge list
                    out_buffer.append((u, v, w, t))

                    if len(out_buffer) >= out_buffer_size:
                        self.__write_out(out_buffer, f_write)
                        out_buffer = []
                if len(out_buffer) > 0:
                    self.__write_out(out_buffer, f_write)
                    out_buffer = []
            print(f"raw edge list created: {raw_data_file_path}")
        else:
            print(f"raw edge list exists: {raw_data_file_path}")

        def __planetoid_node_str_2_id(u: str):
            """dummy function to map node desc to the idx of feature and
            label mat. Planetoid(CORA, CIETEER, PUBMED)'s feat index
            uses node desc (int).

            """

            return int(u)

        # 2: save node feature: pickle of big list of (_u, _feat, _t)
        raw_node_feat_file_path = os_join(
            self.raw_file_path,
            "node_feature_event.pkl",
        )
        if not os_exists(raw_node_feat_file_path):
            # unsorted node feat events
            node_feature_unsorted_events = []
            node_feat_mat = dataset_pyg[0].x.numpy().astype(np.float64)

            if len(node_first_known_seq) != node_feat_mat.shape[0]:
                print(
                    "Warning:"
                    f"nodes in graph {len(node_first_known_seq)}"
                    "!= #node feature"
                    f"({node_feat_mat.shape[0]})."
                    "Remove data folder and try to start from scratch again"
                )

            for u, t in node_first_known_seq:
                # assume: u is idx
                feat = node_feat_mat[__planetoid_node_str_2_id(u), :]
                node_feature_unsorted_events.append((str(u), feat, float(t)))
            with open(raw_node_feat_file_path, "wb") as f_write:
                pickle.dump(node_feature_unsorted_events, f_write)
            print(f"raw node feature created: {raw_node_feat_file_path}")
        else:
            print(f"raw node feature exists: {raw_node_feat_file_path}")

        # 3: save node label: pickle of big list of (_u, _lb, _t)
        raw_node_label_file_path = os_join(
            self.raw_file_path,
            "raw_node_label.pkl",
        )
        if not os_exists(raw_node_label_file_path):
            # unsorted node label events
            node_label_unsorted_events = []
            node_lbs = dataset_pyg[0].y.numpy().astype(np.int16)
            # TODO: output node num_class
            unique_node_class_num = np.unique(node_lbs).shape[0]

            if len(node_first_known_seq) != node_lbs.shape[0]:
                print(
                    "Warning:"
                    f"nodes in graph {len(node_first_known_seq)}"
                    "!= #node labels"
                    f"({node_lbs.shape[0]})."
                    "Remove data folder and try to start from scratch again"
                )

            for u, t in node_first_known_seq:
                lb = node_lbs[__planetoid_node_str_2_id(u)]  #  label as int
                lb_v = np.zeros(unique_node_class_num, dtype=bool)
                lb_v[lb] = True
                node_label_unsorted_events.append((str(u), lb_v, float(t)))
            with open(raw_node_label_file_path, "wb") as f_write:
                pickle.dump(node_label_unsorted_events, f_write)
            print(f"raw node label created: {raw_node_label_file_path}")
        else:
            print(f"raw node label exists: {raw_node_label_file_path}")

    @override
    def parse_edge_event(self, out_buffer_size: int = 10000):
        """parse downloaded raw graph data to edge list, including edge
        timestamp, better to use streaming I/O manner.
        [
            [u: str, v:str, delta_w: float, t:float],
            ...
        ]
        edge_list_file_path = os_join(self.edge_event_path, "data.json")
        """

        raw_data_file_path = os_join(self.raw_file_path, "edge_list.json")
        edge_list_file_path = os_join(self.edge_event_path, "data.json")
        if os.path.exists(edge_list_file_path):
            print(f"unsorted edge list already exists: {edge_list_file_path}")
        else:
            if not os.path.exists(self.edge_event_path):
                if self.verbose:
                    print(
                        f"edge event folder is created: {self.edge_event_path}"
                    )
                os.makedirs(self.edge_event_path)

            out_buffer = []
            with open(edge_list_file_path, "w", encoding="utf-8") as f_write:
                with open(raw_data_file_path, "r", encoding="utf-8") as f_read:
                    for line in f_read:
                        u, v, w, t = json.loads(line.strip())
                        u = str(u)
                        v = str(v)
                        w = float(w)
                        t = float(t)
                        out_buffer.append((u, v, w, t))
                        # NOTE: make it undirected
                        out_buffer.append((v, u, w, t))

                        if len(out_buffer) == out_buffer_size:
                            self.__write_out(out_buffer, f_write)
                            out_buffer = []

                    if len(out_buffer) > 0:
                        self.__write_out(out_buffer, f_write)
                        out_buffer = []
            if self.verbose:
                print(f"edge struct event generated at {edge_list_file_path})")
        return None

    @override
    def parse_node_feature_event(self):
        """convert node feature event (e.g., assign/modify) to list,
        including its timestamp.
        [[ u:str, feature:np.ndarray, t:float ], ...]

        For example: save it to
        node_feat_event_file_path = os_join(
            self.node_feat_event_path, "node_feature_event.pkl"
        )

        """
        node_feat_event_file_path = os_join(
            self.node_feat_event_path, "node_feature_event.pkl"
        )
        if os.path.exists(node_feat_event_file_path):
            print(f"node feature event exists: {node_feat_event_file_path}")
            return None
        if not os.path.exists(self.node_feat_event_path):
            os.makedirs(self.node_feat_event_path)

        raw_node_feat_file_path = os_join(
            self.raw_file_path,
            "node_feature_event.pkl",
        )
        feat_events = []
        with open(raw_node_feat_file_path, "rb") as f:
            raw_node_feature_list = pickle.load(f)

        for i in range(len(raw_node_feature_list)):
            _u = str(raw_node_feature_list[i][0])
            _feat = raw_node_feature_list[i][1].astype(np.float32)
            _t = float(raw_node_feature_list[i][2])
            feat_events.append((_u, _feat, _t))

        with open(node_feat_event_file_path, "wb") as f:
            pickle.dump(feat_events, f)
        print(f"node feature event created: {node_feat_event_file_path}")

    @override
    def parse_node_label_event(self):
        """convert node label event (e.g., assign/modify) to list,
        including its timestamp.
        [[ u:str, label:np.array[Bool], t:float ], ...]

        encode node label into array of int (assuming multi-label
        classification).

        For example: save to
        node_label_event_file_path = os_join(
            self.node_label_event_path, "node_label_event.pkl"
        )
        """
        node_label_event_file_path = os_join(
            self.node_label_event_path, "node_label_event.pkl"
        )
        if os.path.exists(node_label_event_file_path):
            print(f"node label event exists: {node_label_event_file_path}")
            return None

        if not os.path.exists(self.node_label_event_path):
            os.makedirs(self.node_label_event_path)

        raw_node_label_file_path = os_join(
            self.raw_file_path, "raw_node_label.pkl"
        )
        node_label_list = []
        with open(raw_node_label_file_path, "rb") as f:
            raw_node_label_list = pickle.load(f)

        for i in range(len(raw_node_label_list)):
            _u = str(raw_node_label_list[i][0])
            _lb_v: np.ndarray = raw_node_label_list[i][1]
            _t = float(raw_node_label_list[i][2])
            node_label_list.append((_u, _lb_v, _t))

        with open(node_label_event_file_path, "wb") as f:
            pickle.dump(node_label_list, f)
        print(f"node label event created: {node_label_event_file_path}")


class DynGraphReaderFlicker(DynGraphReader):
    def __init__(
        self,
        graph_dataset_name: str,
        local_dataset_dir_abs_path: str,
        graph_event_prefix: str = "snapshot-",
        verbose: bool = False,
        use_undirect: bool = False,
    ):
        super().__init__(
            graph_dataset_name=graph_dataset_name,
            local_dataset_dir_abs_path=local_dataset_dir_abs_path,
            graph_event_prefix=graph_event_prefix,
            verbose=verbose,
        )
        self.use_undirect = use_undirect

    def __write_out(self, buffer, f_write):
        f_write.write("\n".join([json.dumps(_) for _ in buffer]))
        f_write.write("\n")

    @override
    def download_raw_data(self, out_buffer_size: int = 10000):
        """download data to local path

        save to self.node_feat_event_path., ...

        """
        # 1: save edge struct
        dataset_pyg = torch_geometric.datasets.Flickr(
            root=self.raw_file_path,
            transform=T.NormalizeFeatures(),
        )

        node_set = set()
        node_first_known_seq: List[Tuple[str, float]] = []
        if self.use_undirect:
            edge_set = set()

        raw_data_file_path = os_join(self.raw_file_path, "edge_list.json")
        if not os_exists(raw_data_file_path):
            with open(raw_data_file_path, "w") as f_write:
                out_buffer = []
                edge_list = dataset_pyg[0].edge_index.numpy().astype(np.int32)
                _t = 0
                for i in range(edge_list.shape[1]):
                    u = str(edge_list[0, i])
                    v = str(edge_list[1, i])
                    if self.use_undirect:
                        if (u, v) in edge_set or (v, u) in edge_set:
                            continue
                        else:
                            edge_set.add((u, v))
                    w = float(1.0)
                    t = float(_t)
                    _t += 1

                    # fake node time
                    if u not in node_set:
                        node_set.add(u)
                        node_first_known_seq.append((u, t))
                    if v not in node_set:
                        node_set.add(v)
                        node_first_known_seq.append((v, t))

                    # record/write edge list
                    out_buffer.append((u, v, w, t))

                    if len(out_buffer) >= out_buffer_size:
                        self.__write_out(out_buffer, f_write)
                        out_buffer = []
                if len(out_buffer) > 0:
                    self.__write_out(out_buffer, f_write)
                    out_buffer = []
            print(f"raw edge list created: {raw_data_file_path}")
        else:
            print(f"raw edge list exists: {raw_data_file_path}")

        def __planetoid_node_str_2_id(u: str):
            """dummy function to map node desc to the idx of feature and
            label mat. Planetoid(CORA, CIETEER, PUBMED)'s feat index
            uses node desc (int).

            """

            return int(u)

        # 2: save node feature: pickle of big list of (_u, _feat, _t)
        raw_node_feat_file_path = os_join(
            self.raw_file_path,
            "node_feature_event.pkl",
        )
        if not os_exists(raw_node_feat_file_path):
            # unsorted node feat events
            node_feature_unsorted_events = []
            node_feat_mat = dataset_pyg[0].x.numpy().astype(np.float64)

            if len(node_first_known_seq) != node_feat_mat.shape[0]:
                print(
                    "Warning:"
                    f"nodes in graph {len(node_first_known_seq)}"
                    "!= #node feature"
                    f"({node_feat_mat.shape[0]})."
                    "Remove data folder and try to start from scratch again"
                )

            for u, t in node_first_known_seq:
                # assume: u is idx
                feat = node_feat_mat[__planetoid_node_str_2_id(u), :]
                node_feature_unsorted_events.append((str(u), feat, float(t)))
            with open(raw_node_feat_file_path, "wb") as f_write:
                pickle.dump(node_feature_unsorted_events, f_write)
            print(f"raw node feature created: {raw_node_feat_file_path}")
        else:
            print(f"raw node feature exists: {raw_node_feat_file_path}")

        # 3: save node label: pickle of big list of (_u, _lb, _t)
        raw_node_label_file_path = os_join(
            self.raw_file_path,
            "raw_node_label.pkl",
        )
        if not os_exists(raw_node_label_file_path):
            # unsorted node label events
            node_label_unsorted_events = []
            node_lbs = dataset_pyg[0].y.numpy().astype(np.int16)
            # TODO: output node num_class
            unique_node_class_num = np.unique(node_lbs).shape[0]

            if len(node_first_known_seq) != node_lbs.shape[0]:
                print(
                    "Warning:"
                    f"nodes in graph {len(node_first_known_seq)}"
                    "!= #node labels"
                    f"({node_lbs.shape[0]})."
                    "Remove data folder and try to start from scratch again"
                )

            for u, t in node_first_known_seq:
                lb = node_lbs[__planetoid_node_str_2_id(u)]  #  label as int
                lb_v = np.zeros(unique_node_class_num, dtype=bool)
                lb_v[lb] = True
                node_label_unsorted_events.append((str(u), lb_v, float(t)))
            with open(raw_node_label_file_path, "wb") as f_write:
                pickle.dump(node_label_unsorted_events, f_write)
            print(f"raw node label created: {raw_node_label_file_path}")
        else:
            print(f"raw node label exists: {raw_node_label_file_path}")

    @override
    def parse_edge_event(self, out_buffer_size: int = 10000):
        """parse downloaded raw graph data to edge list, including edge
        timestamp, better to use streaming I/O manner.
        [
            [u: str, v:str, delta_w: float, t:float],
            ...
        ]
        edge_list_file_path = os_join(self.edge_event_path, "data.json")
        """

        raw_data_file_path = os_join(self.raw_file_path, "edge_list.json")
        edge_list_file_path = os_join(self.edge_event_path, "data.json")
        if os.path.exists(edge_list_file_path):
            print(f"unsorted edge list already exists: {edge_list_file_path}")
        else:
            if not os.path.exists(self.edge_event_path):
                if self.verbose:
                    print(
                        f"edge event folder is created: {self.edge_event_path}"
                    )
                os.makedirs(self.edge_event_path)

            out_buffer = []
            with open(edge_list_file_path, "w", encoding="utf-8") as f_write:
                with open(raw_data_file_path, "r", encoding="utf-8") as f_read:
                    for line in f_read:
                        u, v, w, t = json.loads(line.strip())
                        u = str(u)
                        v = str(v)
                        w = float(w)
                        t = float(t)
                        out_buffer.append((u, v, w, t))
                        # NOTE: make it undirected
                        out_buffer.append((v, u, w, t))

                        if len(out_buffer) == out_buffer_size:
                            self.__write_out(out_buffer, f_write)
                            out_buffer = []

                    if len(out_buffer) > 0:
                        self.__write_out(out_buffer, f_write)
                        out_buffer = []
            if self.verbose:
                print(f"edge struct event generated at {edge_list_file_path})")
        return None

    @override
    def parse_node_feature_event(self):
        """convert node feature event (e.g., assign/modify) to list,
        including its timestamp.
        [[ u:str, feature:np.ndarray, t:float ], ...]

        For example: save it to
        node_feat_event_file_path = os_join(
            self.node_feat_event_path, "node_feature_event.pkl"
        )

        """
        node_feat_event_file_path = os_join(
            self.node_feat_event_path, "node_feature_event.pkl"
        )
        if os.path.exists(node_feat_event_file_path):
            print(f"node feature event exists: {node_feat_event_file_path}")
            return None
        if not os.path.exists(self.node_feat_event_path):
            os.makedirs(self.node_feat_event_path)

        raw_node_feat_file_path = os_join(
            self.raw_file_path,
            "node_feature_event.pkl",
        )
        feat_events = []
        with open(raw_node_feat_file_path, "rb") as f:
            raw_node_feature_list = pickle.load(f)

        for i in range(len(raw_node_feature_list)):
            _u = str(raw_node_feature_list[i][0])
            _feat = raw_node_feature_list[i][1].astype(np.float32)
            _t = float(raw_node_feature_list[i][2])
            feat_events.append((_u, _feat, _t))

        with open(node_feat_event_file_path, "wb") as f:
            pickle.dump(feat_events, f)
        print(f"node feature event created: {node_feat_event_file_path}")

    @override
    def parse_node_label_event(self):
        """convert node label event (e.g., assign/modify) to list,
        including its timestamp.
        [[ u:str, label:np.array[Bool], t:float ], ...]

        encode node label into array of int (assuming multi-label
        classification).

        For example: save to
        node_label_event_file_path = os_join(
            self.node_label_event_path, "node_label_event.pkl"
        )
        """
        node_label_event_file_path = os_join(
            self.node_label_event_path, "node_label_event.pkl"
        )
        if os.path.exists(node_label_event_file_path):
            print(f"node label event exists: {node_label_event_file_path}")
            return None

        if not os.path.exists(self.node_label_event_path):
            os.makedirs(self.node_label_event_path)

        raw_node_label_file_path = os_join(
            self.raw_file_path, "raw_node_label.pkl"
        )
        node_label_list = []
        with open(raw_node_label_file_path, "rb") as f:
            raw_node_label_list = pickle.load(f)

        for i in range(len(raw_node_label_list)):
            _u = str(raw_node_label_list[i][0])
            _lb_v: np.ndarray = raw_node_label_list[i][1]
            _t = float(raw_node_label_list[i][2])
            node_label_list.append((_u, _lb_v, _t))

        with open(node_label_event_file_path, "wb") as f:
            pickle.dump(node_label_list, f)
        print(f"node label event created: {node_label_event_file_path}")


class DynGraphReaderReddit2(DynGraphReader):
    def __init__(
        self,
        graph_dataset_name: str,
        local_dataset_dir_abs_path: str,
        graph_event_prefix: str = "snapshot-",
        verbose: bool = False,
        use_undirect: bool = False,
    ):
        super().__init__(
            graph_dataset_name=graph_dataset_name,
            local_dataset_dir_abs_path=local_dataset_dir_abs_path,
            graph_event_prefix=graph_event_prefix,
            verbose=verbose,
        )
        self.use_undirect = use_undirect

    def __write_out(self, buffer, f_write):
        f_write.write("\n".join([json.dumps(_) for _ in buffer]))
        f_write.write("\n")

    @override
    def download_raw_data(self, out_buffer_size: int = 10000):
        """download data to local path

        save to self.node_feat_event_path., ...

        """
        # 1: save edge struct
        dataset_pyg = torch_geometric.datasets.Reddit2(
            root=self.raw_file_path,
            transform=T.NormalizeFeatures(),
        )

        node_set = set()
        node_first_known_seq: List[Tuple[str, float]] = []
        if self.use_undirect:
            edge_set = set()

        raw_data_file_path = os_join(self.raw_file_path, "edge_list.json")
        if not os_exists(raw_data_file_path):
            with open(raw_data_file_path, "w") as f_write:
                out_buffer = []
                edge_list = dataset_pyg[0].edge_index.numpy().astype(np.int32)
                _t = 0
                for i in range(edge_list.shape[1]):
                    u = str(edge_list[0, i])
                    v = str(edge_list[1, i])
                    if self.use_undirect:
                        if (u, v) in edge_set or (v, u) in edge_set:
                            continue
                        else:
                            edge_set.add((u, v))
                    w = float(1.0)
                    t = float(_t)
                    _t += 1

                    # fake node time
                    if u not in node_set:
                        node_set.add(u)
                        node_first_known_seq.append((u, t))
                    if v not in node_set:
                        node_set.add(v)
                        node_first_known_seq.append((v, t))

                    # record/write edge list
                    out_buffer.append((u, v, w, t))

                    if len(out_buffer) >= out_buffer_size:
                        self.__write_out(out_buffer, f_write)
                        out_buffer = []
                if len(out_buffer) > 0:
                    self.__write_out(out_buffer, f_write)
                    out_buffer = []
            print(f"raw edge list created: {raw_data_file_path}")
        else:
            print(f"raw edge list exists: {raw_data_file_path}")

        def __planetoid_node_str_2_id(u: str):
            """dummy function to map node desc to the idx of feature and
            label mat. Planetoid(CORA, CIETEER, PUBMED)'s feat index
            uses node desc (int).

            """

            return int(u)

        # 2: save node feature: pickle of big list of (_u, _feat, _t)
        raw_node_feat_file_path = os_join(
            self.raw_file_path,
            "node_feature_event.pkl",
        )
        if not os_exists(raw_node_feat_file_path):
            # unsorted node feat events
            node_feature_unsorted_events = []
            node_feat_mat = dataset_pyg[0].x.numpy().astype(np.float64)

            if len(node_first_known_seq) != node_feat_mat.shape[0]:
                print(
                    "Warning:"
                    f"nodes in graph {len(node_first_known_seq)}"
                    "!= #node feature"
                    f"({node_feat_mat.shape[0]})."
                    "Remove data folder and try to start from scratch again"
                )

            for u, t in node_first_known_seq:
                # assume: u is idx
                feat = node_feat_mat[__planetoid_node_str_2_id(u), :]
                node_feature_unsorted_events.append((str(u), feat, float(t)))
            with open(raw_node_feat_file_path, "wb") as f_write:
                pickle.dump(node_feature_unsorted_events, f_write)
            print(f"raw node feature created: {raw_node_feat_file_path}")
        else:
            print(f"raw node feature exists: {raw_node_feat_file_path}")

        # 3: save node label: pickle of big list of (_u, _lb, _t)
        raw_node_label_file_path = os_join(
            self.raw_file_path,
            "raw_node_label.pkl",
        )
        if not os_exists(raw_node_label_file_path):
            # unsorted node label events
            node_label_unsorted_events = []
            node_lbs = dataset_pyg[0].y.numpy().astype(np.int16)
            # TODO: output node num_class
            unique_node_class_num = np.unique(node_lbs).shape[0]

            if len(node_first_known_seq) != node_lbs.shape[0]:
                print(
                    "Warning:"
                    f"nodes in graph {len(node_first_known_seq)}"
                    "!= #node labels"
                    f"({node_lbs.shape[0]})."
                    "Remove data folder and try to start from scratch again"
                )

            for u, t in node_first_known_seq:
                lb = node_lbs[__planetoid_node_str_2_id(u)]  #  label as int
                lb_v = np.zeros(unique_node_class_num, dtype=bool)
                lb_v[lb] = True
                node_label_unsorted_events.append((str(u), lb_v, float(t)))
            with open(raw_node_label_file_path, "wb") as f_write:
                pickle.dump(node_label_unsorted_events, f_write)
            print(f"raw node label created: {raw_node_label_file_path}")
        else:
            print(f"raw node label exists: {raw_node_label_file_path}")

    @override
    def parse_edge_event(self, out_buffer_size: int = 10000):
        """parse downloaded raw graph data to edge list, including edge
        timestamp, better to use streaming I/O manner.
        [
            [u: str, v:str, delta_w: float, t:float],
            ...
        ]
        edge_list_file_path = os_join(self.edge_event_path, "data.json")
        """

        raw_data_file_path = os_join(self.raw_file_path, "edge_list.json")
        edge_list_file_path = os_join(self.edge_event_path, "data.json")
        if os.path.exists(edge_list_file_path):
            print(f"unsorted edge list already exists: {edge_list_file_path}")
        else:
            if not os.path.exists(self.edge_event_path):
                if self.verbose:
                    print(
                        f"edge event folder is created: {self.edge_event_path}"
                    )
                os.makedirs(self.edge_event_path)

            out_buffer = []
            with open(edge_list_file_path, "w", encoding="utf-8") as f_write:
                with open(raw_data_file_path, "r", encoding="utf-8") as f_read:
                    for line in f_read:
                        u, v, w, t = json.loads(line.strip())
                        u = str(u)
                        v = str(v)
                        w = float(w)
                        t = float(t)
                        out_buffer.append((u, v, w, t))
                        # NOTE: make it undirected
                        out_buffer.append((v, u, w, t))

                        if len(out_buffer) == out_buffer_size:
                            self.__write_out(out_buffer, f_write)
                            out_buffer = []

                    if len(out_buffer) > 0:
                        self.__write_out(out_buffer, f_write)
                        out_buffer = []
            if self.verbose:
                print(f"edge struct event generated at {edge_list_file_path})")
        return None

    @override
    def parse_node_feature_event(self):
        """convert node feature event (e.g., assign/modify) to list,
        including its timestamp.
        [[ u:str, feature:np.ndarray, t:float ], ...]

        For example: save it to
        node_feat_event_file_path = os_join(
            self.node_feat_event_path, "node_feature_event.pkl"
        )

        """
        node_feat_event_file_path = os_join(
            self.node_feat_event_path, "node_feature_event.pkl"
        )
        if os.path.exists(node_feat_event_file_path):
            print(f"node feature event exists: {node_feat_event_file_path}")
            return None
        if not os.path.exists(self.node_feat_event_path):
            os.makedirs(self.node_feat_event_path)

        raw_node_feat_file_path = os_join(
            self.raw_file_path,
            "node_feature_event.pkl",
        )
        feat_events = []
        with open(raw_node_feat_file_path, "rb") as f:
            raw_node_feature_list = pickle.load(f)

        for i in range(len(raw_node_feature_list)):
            _u = str(raw_node_feature_list[i][0])
            _feat = raw_node_feature_list[i][1].astype(np.float32)
            _t = float(raw_node_feature_list[i][2])
            feat_events.append((_u, _feat, _t))

        with open(node_feat_event_file_path, "wb") as f:
            pickle.dump(feat_events, f)
        print(f"node feature event created: {node_feat_event_file_path}")

    @override
    def parse_node_label_event(self):
        """convert node label event (e.g., assign/modify) to list,
        including its timestamp.
        [[ u:str, label:np.array[Bool], t:float ], ...]

        encode node label into array of int (assuming multi-label
        classification).

        For example: save to
        node_label_event_file_path = os_join(
            self.node_label_event_path, "node_label_event.pkl"
        )
        """
        node_label_event_file_path = os_join(
            self.node_label_event_path, "node_label_event.pkl"
        )
        if os.path.exists(node_label_event_file_path):
            print(f"node label event exists: {node_label_event_file_path}")
            return None

        if not os.path.exists(self.node_label_event_path):
            os.makedirs(self.node_label_event_path)

        raw_node_label_file_path = os_join(
            self.raw_file_path, "raw_node_label.pkl"
        )
        node_label_list = []
        with open(raw_node_label_file_path, "rb") as f:
            raw_node_label_list = pickle.load(f)

        for i in range(len(raw_node_label_list)):
            _u = str(raw_node_label_list[i][0])
            _lb_v: np.ndarray = raw_node_label_list[i][1]
            _t = float(raw_node_label_list[i][2])
            node_label_list.append((_u, _lb_v, _t))

        with open(node_label_event_file_path, "wb") as f:
            pickle.dump(node_label_list, f)
        print(f"node label event created: {node_label_event_file_path}")


class DynGraphReaderOGB(DynGraphReader):
    def __init__(
        self,
        graph_dataset_name: str,
        local_dataset_dir_abs_path: str,
        graph_event_prefix: str = "snapshot-",
        verbose: bool = False,
        already_sort: bool = False,
        use_undirect: bool = False,
    ):
        super().__init__(
            graph_dataset_name=graph_dataset_name,
            local_dataset_dir_abs_path=local_dataset_dir_abs_path,
            graph_event_prefix=graph_event_prefix,
            verbose=verbose,
            already_sort=already_sort,
        )
        self.ogbn_root = os_join(
            local_dataset_dir_abs_path,
            "ogbn_datasets",
        )
        self.use_undirect = use_undirect

    def __write_out(self, buffer, f_write):
        f_write.write("\n".join([json.dumps(_) for _ in buffer]))
        f_write.write("\n")

    @override
    def download_raw_data(self, out_buffer_size: int = 10000):
        """download data to local path

        save to self.node_feat_event_path., ...

        """
        # 1: save edge struct

        dataset = NodePropPredDataset(
            name=self.graph_dataset_name,
            root=self.ogbn_root,
        )

        node_set = set()
        node_first_known_seq: List[Tuple[str, float]] = []
        graph, label = dataset[0]

        if self.use_undirect:
            edge_set = set()

        if not os.path.exists(self.raw_file_path):
            os.makedirs(self.raw_file_path)

        raw_data_file_path = os_join(self.raw_file_path, "edge_list.json")
        if not os_exists(raw_data_file_path):
            with open(raw_data_file_path, "w") as f_write:
                out_buffer = []
                edge_list = graph["edge_index"].astype(np.int32)
                i = 0
                for idx in range(edge_list.shape[1]):
                    u = str(edge_list[0, idx])
                    v = str(edge_list[1, idx])
                    if self.use_undirect:
                        if (u, v) in edge_set or (v, u) in edge_set:
                            continue
                        else:
                            edge_set.add((u, v))
                    w = float(1.0)
                    t = float(i)
                    i += 1

                    # fake node time
                    if u not in node_set:
                        node_set.add(u)
                        node_first_known_seq.append((u, t))
                    if v not in node_set:
                        node_set.add(v)
                        node_first_known_seq.append((v, t))

                    # record/write edge list
                    out_buffer.append((u, v, w, t))

                    if len(out_buffer) >= out_buffer_size:
                        self.__write_out(out_buffer, f_write)
                        out_buffer = []
                if len(out_buffer) > 0:
                    self.__write_out(out_buffer, f_write)
                    out_buffer = []
            print(f"raw edge list created: {raw_data_file_path}")
        else:
            print(f"raw edge list exists: {raw_data_file_path}")

        def __ogb_node_str_2_id(u: str):
            """dummy function to map node desc to the idx of feature and
            label mat. OGB's feat index
            uses node desc (int).

            """

            return int(u)

        # 2: save node feature: pickle of big list of (_u, _feat, _t)
        raw_node_feat_file_path = os_join(
            self.raw_file_path,
            "node_feature_event.pkl",
        )
        if not os_exists(raw_node_feat_file_path):
            # unsorted node feat events
            node_feature_unsorted_events = []
            node_feat_mat = graph["node_feat"].astype(np.float32)

            if len(node_first_known_seq) != node_feat_mat.shape[0]:
                print(
                    "Warning:"
                    f"nodes in graph {len(node_first_known_seq)}"
                    "!= #node feature"
                    f"({node_feat_mat.shape[0]})."
                    "Remove data folder and try to start from scratch again"
                )

            for u, t in node_first_known_seq:
                # assume: u is idx
                feat = node_feat_mat[__ogb_node_str_2_id(u), :]
                node_feature_unsorted_events.append((str(u), feat, float(t)))
            with open(raw_node_feat_file_path, "wb") as f_write:
                pickle.dump(node_feature_unsorted_events, f_write)
            print(f"raw node feature created: {raw_node_feat_file_path}")
        else:
            print(f"raw node feature exists: {raw_node_feat_file_path}")

        # 3: save node label: pickle of big list of (_u, _lb, _t)
        raw_node_label_file_path = os_join(
            self.raw_file_path,
            "raw_node_label.pkl",
        )
        if not os_exists(raw_node_label_file_path):
            # unsorted node label events
            node_label_unsorted_events = []
            node_lbs = label.astype(np.int16)
            # TODO: output node num_class
            unique_node_class_num = np.unique(node_lbs).shape[0]

            if len(node_first_known_seq) != node_lbs.shape[0]:
                print(
                    "Warning:"
                    f"nodes in graph {len(node_first_known_seq)}"
                    "!= #node labels"
                    f"({node_lbs.shape[0]})."
                    "Remove data folder and try to start from scratch again"
                )

            for u, t in node_first_known_seq:
                lb = node_lbs[__ogb_node_str_2_id(u)]  #  label as int
                lb_v = np.zeros(unique_node_class_num, dtype=bool)
                lb_v[lb] = True
                node_label_unsorted_events.append((str(u), lb_v, float(t)))
            with open(raw_node_label_file_path, "wb") as f_write:
                pickle.dump(node_label_unsorted_events, f_write)
            print(f"raw node label created: {raw_node_label_file_path}")
        else:
            print(f"raw node label exists: {raw_node_label_file_path}")

    @override
    def parse_edge_event(self, out_buffer_size: int = 10000):
        """parse downloaded raw graph data to edge list, including edge
        timestamp, better to use streaming I/O manner.
        [
            [u: str, v:str, delta_w: float, t:float],
            ...
        ]
        edge_list_file_path = os_join(self.edge_event_path, "data.json")
        """

        raw_data_file_path = os_join(self.raw_file_path, "edge_list.json")
        edge_list_file_path = os_join(self.edge_event_path, "data.json")
        if os.path.exists(edge_list_file_path):
            print(f"unsorted edge list already exists: {edge_list_file_path}")
        else:
            if not os.path.exists(self.edge_event_path):
                if self.verbose:
                    print(
                        f"edge event folder is created: {self.edge_event_path}"
                    )
                os.makedirs(self.edge_event_path)

            out_buffer = []
            with open(edge_list_file_path, "w", encoding="utf-8") as f_write:
                with open(raw_data_file_path, "r", encoding="utf-8") as f_read:
                    for line in f_read:
                        u, v, w, t = json.loads(line.strip())
                        u = str(u)
                        v = str(v)
                        w = float(w)
                        t = float(t)
                        out_buffer.append((u, v, w, t))
                        # NOTE: make it undirected
                        out_buffer.append((v, u, w, t))

                        if len(out_buffer) == out_buffer_size:
                            self.__write_out(out_buffer, f_write)
                            out_buffer = []

                    if len(out_buffer) > 0:
                        self.__write_out(out_buffer, f_write)
                        out_buffer = []
            if self.verbose:
                print(f"edge struct event generated at {edge_list_file_path})")
        return None

    @override
    def parse_node_feature_event(self):
        """convert node feature event (e.g., assign/modify) to list,
        including its timestamp.
        [[ u:str, feature:np.ndarray, t:float ], ...]

        For example: save it to
        node_feat_event_file_path = os_join(
            self.node_feat_event_path, "node_feature_event.pkl"
        )

        """
        node_feat_event_file_path = os_join(
            self.node_feat_event_path, "node_feature_event.pkl"
        )
        if os.path.exists(node_feat_event_file_path):
            print(f"node feature event exists: {node_feat_event_file_path}")
            return None
        if not os.path.exists(self.node_feat_event_path):
            os.makedirs(self.node_feat_event_path)

        raw_node_feat_file_path = os_join(
            self.raw_file_path,
            "node_feature_event.pkl",
        )
        feat_events = []
        with open(raw_node_feat_file_path, "rb") as f:
            raw_node_feature_list = pickle.load(f)

        for i in range(len(raw_node_feature_list)):
            _u = str(raw_node_feature_list[i][0])
            _feat = raw_node_feature_list[i][1].astype(np.float32)
            _t = float(raw_node_feature_list[i][2])
            feat_events.append((_u, _feat, _t))

        with open(node_feat_event_file_path, "wb") as f:
            pickle.dump(feat_events, f)
        print(f"node feature event created: {node_feat_event_file_path}")

    @override
    def parse_node_label_event(self):
        """convert node label event (e.g., assign/modify) to list,
        including its timestamp.
        [[ u:str, label:np.array[Bool], t:float ], ...]

        encode node label into array of int (assuming multi-label
        classification).

        For example: save to
        node_label_event_file_path = os_join(
            self.node_label_event_path, "node_label_event.pkl"
        )
        """
        node_label_event_file_path = os_join(
            self.node_label_event_path, "node_label_event.pkl"
        )
        if os.path.exists(node_label_event_file_path):
            print(f"node label event exists: {node_label_event_file_path}")
            return None

        if not os.path.exists(self.node_label_event_path):
            os.makedirs(self.node_label_event_path)

        raw_node_label_file_path = os_join(
            self.raw_file_path, "raw_node_label.pkl"
        )
        node_label_list = []
        with open(raw_node_label_file_path, "rb") as f:
            raw_node_label_list = pickle.load(f)

        for i in range(len(raw_node_label_list)):
            _u = str(raw_node_label_list[i][0])
            _lb_v: np.ndarray = raw_node_label_list[i][1]
            _t = float(raw_node_label_list[i][2])
            node_label_list.append((_u, _lb_v, _t))

        with open(node_label_event_file_path, "wb") as f:
            pickle.dump(node_label_list, f)
        print(f"node label event created: {node_label_event_file_path}")


class DynGraphReaderHeterophilyActor(DynGraphReader):
    def __init__(
        self,
        graph_dataset_name: str,
        local_dataset_dir_abs_path: str,
        graph_event_prefix: str = "snapshot-",
        verbose: bool = False,
    ):
        super().__init__(
            graph_dataset_name=graph_dataset_name,
            local_dataset_dir_abs_path=local_dataset_dir_abs_path,
            graph_event_prefix=graph_event_prefix,
            verbose=verbose,
        )

    def __write_out(self, buffer, f_write):
        f_write.write("\n".join([json.dumps(_) for _ in buffer]))
        f_write.write("\n")

    @override
    def download_raw_data(self, out_buffer_size: int = 10000):
        """download data to local path

        save to self.node_feat_event_path., ...

        """
        # 1: save edge struct

        if os.path.exists(self.raw_file_path) is not True:
            os.makedirs(self.raw_file_path)

        dataset_pyg = torch_geometric.datasets.Actor(
            root=self.raw_file_path,
            transform=T.NormalizeFeatures(),
        )

        node_set = set()
        node_first_known_seq: List[Tuple[str, float]] = []

        raw_data_file_path = os_join(self.raw_file_path, "edge_list.json")
        if not os_exists(raw_data_file_path):
            with open(raw_data_file_path, "w") as f_write:
                out_buffer = []
                edge_list = dataset_pyg[0].edge_index.numpy().astype(np.int32)
                for i in range(edge_list.shape[1]):
                    u = str(edge_list[0, i])
                    v = str(edge_list[1, i])
                    w = float(1.0)
                    t = float(i)

                    # fake node time
                    if u not in node_set:
                        node_set.add(u)
                        node_first_known_seq.append((u, t))
                    if v not in node_set:
                        node_set.add(v)
                        node_first_known_seq.append((v, t))

                    # record/write edge list
                    out_buffer.append((u, v, w, t))

                    if len(out_buffer) >= out_buffer_size:
                        self.__write_out(out_buffer, f_write)
                        out_buffer = []
                if len(out_buffer) > 0:
                    self.__write_out(out_buffer, f_write)
                    out_buffer = []
            print(f"raw edge list created: {raw_data_file_path}")
        else:
            print(f"raw edge list exists: {raw_data_file_path}")

        def __planetoid_node_str_2_id(u: str):
            """dummy function to map node desc to the idx of feature and
            label mat. Planetoid(CORA, CIETEER, PUBMED)'s feat index
            uses node desc (int).

            """

            return int(u)

        # 2: save node feature: pickle of big list of (_u, _feat, _t)
        raw_node_feat_file_path = os_join(
            self.raw_file_path,
            "node_feature_event.pkl",
        )
        if not os_exists(raw_node_feat_file_path):
            # unsorted node feat events
            node_feature_unsorted_events = []
            node_feat_mat = dataset_pyg[0].x.numpy().astype(np.float64)

            if len(node_first_known_seq) != node_feat_mat.shape[0]:
                print(
                    "Warning:"
                    f"nodes in graph {len(node_first_known_seq)}"
                    "!= #node feature"
                    f"({node_feat_mat.shape[0]})."
                    "Remove data folder and try to start from scratch again"
                )

            for u, t in node_first_known_seq:
                # assume: u is idx
                feat = node_feat_mat[__planetoid_node_str_2_id(u), :]
                node_feature_unsorted_events.append((str(u), feat, float(t)))
            with open(raw_node_feat_file_path, "wb") as f_write:
                pickle.dump(node_feature_unsorted_events, f_write)
            print(f"raw node feature created: {raw_node_feat_file_path}")
        else:
            print(f"raw node feature exists: {raw_node_feat_file_path}")

        # 3: save node label: pickle of big list of (_u, _lb, _t)
        raw_node_label_file_path = os_join(
            self.raw_file_path,
            "raw_node_label.pkl",
        )
        if not os_exists(raw_node_label_file_path):
            # unsorted node label events
            node_label_unsorted_events = []
            node_lbs = dataset_pyg[0].y.numpy().astype(np.int16)
            # TODO: output node num_class
            unique_node_class_num = np.unique(node_lbs).shape[0]

            if len(node_first_known_seq) != node_lbs.shape[0]:
                print(
                    "Warning:"
                    f"nodes in graph {len(node_first_known_seq)}"
                    "!= #node labels"
                    f"({node_lbs.shape[0]})."
                    "Remove data folder and try to start from scratch again"
                )

            for u, t in node_first_known_seq:
                lb = node_lbs[__planetoid_node_str_2_id(u)]  #  label as int
                lb_v = np.zeros(unique_node_class_num, dtype=bool)
                lb_v[lb] = True
                node_label_unsorted_events.append((str(u), lb_v, float(t)))
            with open(raw_node_label_file_path, "wb") as f_write:
                pickle.dump(node_label_unsorted_events, f_write)
            print(f"raw node label created: {raw_node_label_file_path}")
        else:
            print(f"raw node label exists: {raw_node_label_file_path}")

    @override
    def parse_edge_event(self, out_buffer_size: int = 10000):
        """parse downloaded raw graph data to edge list, including edge
        timestamp, better to use streaming I/O manner.
        [
            [u: str, v:str, delta_w: float, t:float],
            ...
        ]
        edge_list_file_path = os_join(self.edge_event_path, "data.json")
        """

        raw_data_file_path = os_join(self.raw_file_path, "edge_list.json")
        edge_list_file_path = os_join(self.edge_event_path, "data.json")
        if os.path.exists(edge_list_file_path):
            print(f"unsorted edge list already exists: {edge_list_file_path}")
        else:
            if not os.path.exists(self.edge_event_path):
                if self.verbose:
                    print(
                        f"edge event folder is created: {self.edge_event_path}"
                    )
                os.makedirs(self.edge_event_path)

            out_buffer = []
            with open(edge_list_file_path, "w", encoding="utf-8") as f_write:
                with open(raw_data_file_path, "r", encoding="utf-8") as f_read:
                    for line in f_read:
                        u, v, w, t = json.loads(line.strip())
                        u = str(u)
                        v = str(v)
                        w = float(w)
                        t = float(t)
                        out_buffer.append((u, v, w, t))
                        # NOTE: make it undirected
                        out_buffer.append((v, u, w, t))

                        if len(out_buffer) == out_buffer_size:
                            self.__write_out(out_buffer, f_write)
                            out_buffer = []

                    if len(out_buffer) > 0:
                        self.__write_out(out_buffer, f_write)
                        out_buffer = []
            if self.verbose:
                print(f"edge struct event generated at {edge_list_file_path})")
        return None

    @override
    def parse_node_feature_event(self):
        """convert node feature event (e.g., assign/modify) to list,
        including its timestamp.
        [[ u:str, feature:np.ndarray, t:float ], ...]

        For example: save it to
        node_feat_event_file_path = os_join(
            self.node_feat_event_path, "node_feature_event.pkl"
        )

        """
        node_feat_event_file_path = os_join(
            self.node_feat_event_path, "node_feature_event.pkl"
        )
        if os.path.exists(node_feat_event_file_path):
            print(f"node feature event exists: {node_feat_event_file_path}")
            return None
        if not os.path.exists(self.node_feat_event_path):
            os.makedirs(self.node_feat_event_path)

        raw_node_feat_file_path = os_join(
            self.raw_file_path,
            "node_feature_event.pkl",
        )
        feat_events = []
        with open(raw_node_feat_file_path, "rb") as f:
            raw_node_feature_list = pickle.load(f)

        for i in range(len(raw_node_feature_list)):
            _u = str(raw_node_feature_list[i][0])
            _feat = raw_node_feature_list[i][1].astype(np.float32)
            _t = float(raw_node_feature_list[i][2])
            feat_events.append((_u, _feat, _t))

        with open(node_feat_event_file_path, "wb") as f:
            pickle.dump(feat_events, f)
        print(f"node feature event created: {node_feat_event_file_path}")

    @override
    def parse_node_label_event(self):
        """convert node label event (e.g., assign/modify) to list,
        including its timestamp.
        [[ u:str, label:np.array[Bool], t:float ], ...]

        encode node label into array of int (assuming multi-label
        classification).

        For example: save to
        node_label_event_file_path = os_join(
            self.node_label_event_path, "node_label_event.pkl"
        )
        """
        node_label_event_file_path = os_join(
            self.node_label_event_path, "node_label_event.pkl"
        )
        if os.path.exists(node_label_event_file_path):
            print(f"node label event exists: {node_label_event_file_path}")
            return None

        if not os.path.exists(self.node_label_event_path):
            os.makedirs(self.node_label_event_path)

        raw_node_label_file_path = os_join(
            self.raw_file_path, "raw_node_label.pkl"
        )
        node_label_list = []
        with open(raw_node_label_file_path, "rb") as f:
            raw_node_label_list = pickle.load(f)

        for i in range(len(raw_node_label_list)):
            _u = str(raw_node_label_list[i][0])
            _lb_v: np.ndarray = raw_node_label_list[i][1]
            _t = float(raw_node_label_list[i][2])
            node_label_list.append((_u, _lb_v, _t))

        with open(node_label_event_file_path, "wb") as f:
            pickle.dump(node_label_list, f)
        print(f"node label event created: {node_label_event_file_path}")


class DynGraphReaderHeterophilyWiki(DynGraphReader):
    def __init__(
        self,
        graph_dataset_name: str,
        local_dataset_dir_abs_path: str,
        graph_event_prefix: str = "snapshot-",
        verbose: bool = False,
    ):
        super().__init__(
            graph_dataset_name=graph_dataset_name,
            local_dataset_dir_abs_path=local_dataset_dir_abs_path,
            graph_event_prefix=graph_event_prefix,
            verbose=verbose,
        )

    def __write_out(self, buffer, f_write):
        f_write.write("\n".join([json.dumps(_) for _ in buffer]))
        f_write.write("\n")

    @override
    def download_raw_data(self, out_buffer_size: int = 10000):
        """download data to local path

        save to self.node_feat_event_path., ...

        """
        # 1: save edge struct

        if os.path.exists(self.raw_file_path) is not True:
            os.makedirs(self.raw_file_path)

        dataset_pyg = torch_geometric.datasets.WikipediaNetwork(
            root=self.raw_file_path,
            name=self.graph_dataset_name,
            transform=T.NormalizeFeatures(),
        )

        node_set = set()
        node_first_known_seq: List[Tuple[str, float]] = []

        raw_data_file_path = os_join(self.raw_file_path, "edge_list.json")
        if not os_exists(raw_data_file_path):
            with open(raw_data_file_path, "w") as f_write:
                out_buffer = []
                edge_list = dataset_pyg[0].edge_index.numpy().astype(np.int32)
                for i in range(edge_list.shape[1]):
                    u = str(edge_list[0, i])
                    v = str(edge_list[1, i])
                    w = float(1.0)
                    t = float(i)

                    # fake node time
                    if u not in node_set:
                        node_set.add(u)
                        node_first_known_seq.append((u, t))
                    if v not in node_set:
                        node_set.add(v)
                        node_first_known_seq.append((v, t))

                    # record/write edge list
                    out_buffer.append((u, v, w, t))

                    if len(out_buffer) >= out_buffer_size:
                        self.__write_out(out_buffer, f_write)
                        out_buffer = []
                if len(out_buffer) > 0:
                    self.__write_out(out_buffer, f_write)
                    out_buffer = []
            print(f"raw edge list created: {raw_data_file_path}")
        else:
            print(f"raw edge list exists: {raw_data_file_path}")

        def __planetoid_node_str_2_id(u: str):
            """dummy function to map node desc to the idx of feature and
            label mat. Planetoid(CORA, CIETEER, PUBMED)'s feat index
            uses node desc (int).

            """

            return int(u)

        # 2: save node feature: pickle of big list of (_u, _feat, _t)
        raw_node_feat_file_path = os_join(
            self.raw_file_path,
            "node_feature_event.pkl",
        )
        if not os_exists(raw_node_feat_file_path):
            # unsorted node feat events
            node_feature_unsorted_events = []
            node_feat_mat = dataset_pyg[0].x.numpy().astype(np.float64)

            if len(node_first_known_seq) != node_feat_mat.shape[0]:
                print(
                    "Warning:"
                    f"nodes in graph {len(node_first_known_seq)}"
                    "!= #node feature"
                    f"({node_feat_mat.shape[0]})."
                    "Remove data folder and try to start from scratch again"
                )

            for u, t in node_first_known_seq:
                # assume: u is idx
                feat = node_feat_mat[__planetoid_node_str_2_id(u), :]
                node_feature_unsorted_events.append((str(u), feat, float(t)))
            with open(raw_node_feat_file_path, "wb") as f_write:
                pickle.dump(node_feature_unsorted_events, f_write)
            print(f"raw node feature created: {raw_node_feat_file_path}")
        else:
            print(f"raw node feature exists: {raw_node_feat_file_path}")

        # 3: save node label: pickle of big list of (_u, _lb, _t)
        raw_node_label_file_path = os_join(
            self.raw_file_path,
            "raw_node_label.pkl",
        )
        if not os_exists(raw_node_label_file_path):
            # unsorted node label events
            node_label_unsorted_events = []
            node_lbs = dataset_pyg[0].y.numpy().astype(np.int16)
            # TODO: output node num_class
            unique_node_class_num = np.unique(node_lbs).shape[0]

            if len(node_first_known_seq) != node_lbs.shape[0]:
                print(
                    "Warning:"
                    f"nodes in graph {len(node_first_known_seq)}"
                    "!= #node labels"
                    f"({node_lbs.shape[0]})."
                    "Remove data folder and try to start from scratch again"
                )

            for u, t in node_first_known_seq:
                lb = node_lbs[__planetoid_node_str_2_id(u)]  #  label as int
                lb_v = np.zeros(unique_node_class_num, dtype=bool)
                lb_v[lb] = True
                node_label_unsorted_events.append((str(u), lb_v, float(t)))
            with open(raw_node_label_file_path, "wb") as f_write:
                pickle.dump(node_label_unsorted_events, f_write)
            print(f"raw node label created: {raw_node_label_file_path}")
        else:
            print(f"raw node label exists: {raw_node_label_file_path}")

    @override
    def parse_edge_event(self, out_buffer_size: int = 10000):
        """parse downloaded raw graph data to edge list, including edge
        timestamp, better to use streaming I/O manner.
        [
            [u: str, v:str, delta_w: float, t:float],
            ...
        ]
        edge_list_file_path = os_join(self.edge_event_path, "data.json")
        """

        raw_data_file_path = os_join(self.raw_file_path, "edge_list.json")
        edge_list_file_path = os_join(self.edge_event_path, "data.json")
        if os.path.exists(edge_list_file_path):
            print(f"unsorted edge list already exists: {edge_list_file_path}")
        else:
            if not os.path.exists(self.edge_event_path):
                if self.verbose:
                    print(
                        f"edge event folder is created: {self.edge_event_path}"
                    )
                os.makedirs(self.edge_event_path)

            out_buffer = []
            with open(edge_list_file_path, "w", encoding="utf-8") as f_write:
                with open(raw_data_file_path, "r", encoding="utf-8") as f_read:
                    for line in f_read:
                        u, v, w, t = json.loads(line.strip())
                        u = str(u)
                        v = str(v)
                        w = float(w)
                        t = float(t)
                        out_buffer.append((u, v, w, t))
                        # NOTE: make it undirected
                        out_buffer.append((v, u, w, t))

                        if len(out_buffer) == out_buffer_size:
                            self.__write_out(out_buffer, f_write)
                            out_buffer = []

                    if len(out_buffer) > 0:
                        self.__write_out(out_buffer, f_write)
                        out_buffer = []
            if self.verbose:
                print(f"edge struct event generated at {edge_list_file_path})")
        return None

    @override
    def parse_node_feature_event(self):
        """convert node feature event (e.g., assign/modify) to list,
        including its timestamp.
        [[ u:str, feature:np.ndarray, t:float ], ...]

        For example: save it to
        node_feat_event_file_path = os_join(
            self.node_feat_event_path, "node_feature_event.pkl"
        )

        """
        node_feat_event_file_path = os_join(
            self.node_feat_event_path, "node_feature_event.pkl"
        )
        if os.path.exists(node_feat_event_file_path):
            print(f"node feature event exists: {node_feat_event_file_path}")
            return None
        if not os.path.exists(self.node_feat_event_path):
            os.makedirs(self.node_feat_event_path)

        raw_node_feat_file_path = os_join(
            self.raw_file_path,
            "node_feature_event.pkl",
        )
        feat_events = []
        with open(raw_node_feat_file_path, "rb") as f:
            raw_node_feature_list = pickle.load(f)

        for i in range(len(raw_node_feature_list)):
            _u = str(raw_node_feature_list[i][0])
            _feat = raw_node_feature_list[i][1].astype(np.float32)
            _t = float(raw_node_feature_list[i][2])
            feat_events.append((_u, _feat, _t))

        with open(node_feat_event_file_path, "wb") as f:
            pickle.dump(feat_events, f)
        print(f"node feature event created: {node_feat_event_file_path}")

    @override
    def parse_node_label_event(self):
        """convert node label event (e.g., assign/modify) to list,
        including its timestamp.
        [[ u:str, label:np.array[Bool], t:float ], ...]

        encode node label into array of int (assuming multi-label
        classification).

        For example: save to
        node_label_event_file_path = os_join(
            self.node_label_event_path, "node_label_event.pkl"
        )
        """
        node_label_event_file_path = os_join(
            self.node_label_event_path, "node_label_event.pkl"
        )
        if os.path.exists(node_label_event_file_path):
            print(f"node label event exists: {node_label_event_file_path}")
            return None

        if not os.path.exists(self.node_label_event_path):
            os.makedirs(self.node_label_event_path)

        raw_node_label_file_path = os_join(
            self.raw_file_path, "raw_node_label.pkl"
        )
        node_label_list = []
        with open(raw_node_label_file_path, "rb") as f:
            raw_node_label_list = pickle.load(f)

        for i in range(len(raw_node_label_list)):
            _u = str(raw_node_label_list[i][0])
            _lb_v: np.ndarray = raw_node_label_list[i][1]
            _t = float(raw_node_label_list[i][2])
            node_label_list.append((_u, _lb_v, _t))

        with open(node_label_event_file_path, "wb") as f:
            pickle.dump(node_label_list, f)
        print(f"node label event created: {node_label_event_file_path}")
