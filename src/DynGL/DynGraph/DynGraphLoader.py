from typing import List, Any
from dataclasses import dataclass
from dataclasses_json import dataclass_json

import numpy as np
from os.path import join as os_join
import os
import json
import pickle


@dataclass_json
@dataclass
class DynGraphMetadata:
    # total snapshot
    total_snapshot: int
    # total nodes in all snapshots
    total_num_nodes: int
    # the interval timestamp unit between two snapshots
    snapshot_interval: float
    # the start timestamp unit for the unit
    base_snapshot_t: Any

    # the first/last timestamp of edge event.
    edge_event_timestamp_max: float
    edge_event_timestamp_min: float

    # node feature dim
    node_feat_dim: int
    # node label class num (assume it is multi-label)
    node_label_num_class: int

    # list of edge/node events in each snapshot
    num_edge_events_per_snapshot: List[int]
    num_node_per_snapshot: List[int]
    num_node_feature_events_per_snapshot: List[int]
    num_node_label_events_per_snapshot: List[int]

    edge_event_split_path_list: List[str]
    node_feature_event_split_path_list: List[str]
    node_label_snap_file_path_list: List[str]

    def __str__(self):
        string_list: List[str] = []
        string_list.append(f"Total snapshots:\t{self.total_snapshot}")
        string_list.append(f"Base Snapshot-t:\t{self.base_snapshot_t}")
        string_list.append(f"Snapshot-interval:\t{self.snapshot_interval}")

        string_list.append(f"Max timestamp:\t{self.edge_event_timestamp_max}")
        string_list.append(f"Min timestamp:\t{self.edge_event_timestamp_min}")
        string_list.append(f"Total nodes:\t{self.total_num_nodes}")
        string_list.append(
            f"Total edge events:\t{np.sum(self.num_edge_events_per_snapshot)}"
        )
        string_list.append(
            "Total node feature events:\t"
            f"{np.sum(self.num_node_feature_events_per_snapshot)}"
        )
        string_list.append(
            "Total node label events:\t"
            f"{np.sum(self.num_node_label_events_per_snapshot)}"
        )

        string_list.append(
            "File path for edge struct snapshots:\t"
            f"{len(self.edge_event_split_path_list)}"
        )
        string_list.append(
            "File path for node feature snapshots:\t"
            f"{len(self.node_feature_event_split_path_list)}"
        )
        string_list.append(
            "File path for node label snapshots:\t"
            f"{len(self.node_label_snap_file_path_list)}"
        )
        return "\n".join(string_list)


class DynGraphReader:
    def __init__(
        self,
        graph_dataset_name: str,
        local_dataset_dir_abs_path: str,
        graph_event_prefix: str = "snapshot-",
        verbose: bool = False,
        already_sort: bool = False,
    ):
        self.graph_dataset_name = graph_dataset_name
        self.graph_event_prefix = graph_event_prefix
        self.already_sort = already_sort

        self.local_graph_dataset_dir_abs_path = os_join(
            local_dataset_dir_abs_path, self.graph_dataset_name
        )

        self.raw_file_path: str = os_join(
            self.local_graph_dataset_dir_abs_path, "raw_data"
        )
        self.edge_event_path: str = os_join(
            self.local_graph_dataset_dir_abs_path, "edge_event"
        )
        self.edge_event_sorted_path: str = os_join(
            self.local_graph_dataset_dir_abs_path, "edge_event_sorted"
        )

        self.node_feat_event_path: str = os_join(
            self.local_graph_dataset_dir_abs_path, "node_feature_event"
        )
        self.node_feat_event_sorted_path: str = os_join(
            self.local_graph_dataset_dir_abs_path, "node_feature_event_sorted"
        )

        self.node_label_event_path: str = os_join(
            self.local_graph_dataset_dir_abs_path, "node_label_event"
        )
        self.node_label_event_sorted_path: str = os_join(
            self.local_graph_dataset_dir_abs_path, "node_label_event_sorted"
        )

        self.edge_event_split_path: str = os_join(
            self.local_graph_dataset_dir_abs_path, "edge_event_snapshots"
        )
        self.node_feature_event_split_path: str = os_join(
            self.local_graph_dataset_dir_abs_path,
            "node_feature_event_snapshots",
        )
        self.node_label_event_split_path: str = os_join(
            self.local_graph_dataset_dir_abs_path, "node_label_event_snapshots"
        )

        self.verbose = verbose

    def download_raw_data(self):
        """download data to local path

        save to self.node_feat_event_path

        """
        raise NotImplementedError

    def parse_edge_event(self):
        """parse downloaded raw graph data to edge list, including edge
        timestamp, better to use streaming I/O manner.
        [
            [u: str, v:str, delta_w: float, t:float],
            ...
        ]
        edge_list_file_path = os_join(self.edge_event_path, "data.json")
        """

        raise NotImplementedError

    def sort_edge_event(self, out_buffer_size: int = 10000):
        """sorted the parse edge list, save to local path.
        Note: It may be a memory-bound task for sorting a big arr, e.g.,
        sorting billion edges.

        Merge sort is recommanded

        Some downloaded edge-list is already sorted.

        save to self.edge_event_sorted_path

        """

        def __write_out(buffer, f_write):
            f_write.write("\n".join([json.dumps(_) for _ in buffer]))
            f_write.write("\n")

        edge_list_sorted_file_path = os_join(
            self.edge_event_sorted_path, "data.json"
        )
        if os.path.exists(edge_list_sorted_file_path):
            if self.verbose:
                print(
                    "sorted edge event already exists at                     "
                    f"    {edge_list_sorted_file_path}"
                )
        else:
            edge_list_file_path = os_join(self.edge_event_path, "data.json")
            buffer_read = []
            with open(edge_list_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line.strip())
                    buffer_read.append(d)
                    if len(buffer_read) > 1e6 and len(buffer_read) % 1000 == 0:
                        print(
                            f"curt #unsorted edges:{len(buffer_read)}",
                            end="\r",
                        )

            # sort
            # d = (u, v, w, t)
            # memory in-efficient
            # sorted_buf = sorted(buffer_read, key=lambda x: x[3])
            # in-place to save memory
            if not self.already_sort:
                print("sorting edges, it may need a lot of memory")
                buffer_read.sort(key=lambda x: x[3])
            print(f"sorted edge: {len(buffer_read)}")
            sorted_buf = buffer_read

            if not os.path.exists(self.edge_event_sorted_path):
                if self.verbose:
                    print(
                        "edge event sorted dir was created at                "
                        f"             {self.edge_event_sorted_path}"
                    )
                os.makedirs(self.edge_event_sorted_path)
            out_buffer = []
            with open(
                edge_list_sorted_file_path, "w", encoding="utf-8"
            ) as f_write:
                for d in sorted_buf:
                    out_buffer.append(d)

                    if len(out_buffer) == out_buffer_size:
                        __write_out(out_buffer, f_write)
                        out_buffer = []

                if len(out_buffer) > 0:
                    __write_out(out_buffer, f_write)
                    out_buffer = []

    def parse_node_feature_event(self):
        """convert node feature event (e.g., assign/modify) to list,
        including its timestamp.
        [[ u:str, feature:np.ndarray, t:float ], ...]

        For example: save it to
        node_feat_event_file_path = os_join(
            self.node_feat_event_path, "node_feature_event.pkl"
        )

        """

        raise NotImplementedError

    def sort_node_feature_event(self):
        """sort node feature event (e.g., assign/modify) by timestamp.

        save to self.node_feat_event_sorted_path

        # [(u:str, feat_v:np.ndarray, t:float), ...]
        # self.node_feat_event_sorted_path / node_feature_event_sorted.pkl

        """
        node_feat_event_file_path = os_join(
            self.node_feat_event_path, "node_feature_event.pkl"
        )
        assert os.path.exists(node_feat_event_file_path), (
            "node feature event (unsorted) does not exist at"
            f"{node_feat_event_file_path}"
        )

        node_feat_event_sorted_file_path = os_join(
            self.node_feat_event_sorted_path, "node_feature_event_sorted.pkl"
        )
        if os.path.exists(node_feat_event_sorted_file_path):
            if self.verbose:
                print(
                    "sorted node feature already exists at: "
                    f"{node_feat_event_sorted_file_path}"
                )
        else:
            if not os.path.exists(self.node_feat_event_sorted_path):
                os.makedirs(self.node_feat_event_sorted_path)

            node_feat_event_file: List[Any] = None
            with open(node_feat_event_file_path, "rb") as f:
                node_feat_event_file = pickle.load(f)

            # sorted_buf = [(u:str, feat_v:np.ndarray, t:float), ...]
            # sorted_buf = sorted(node_feat_event_file, key=lambda x: x[2])
            node_feat_event_file.sort(key=lambda x: x[2])
            sorted_buf = node_feat_event_file

            with open(node_feat_event_sorted_file_path, "wb") as f:
                pickle.dump(sorted_buf, f)
        return None

    def parse_node_label_event(self):
        """convert node label event (e.g., assign/modify) to list,
        including its timestamp.
        [[ u:str, label:np.array[Bool], t:float ], ...]

        encode node label into array of bool (assuming multi-label
        classification).

        For example: save to
        node_label_event_file_path = os_join(
            self.node_label_event_path, "node_label_event.pkl"
        )

        """

        raise NotImplementedError

    def sort_node_label_event(self):
        """sort node feature event (e.g., assign/modify) by timestamp.

        [ u:str, label:int, t:float ], ...

        save to self.node_label_event_path
        """
        node_label_event_file_path = os_join(
            self.node_label_event_path, "node_label_event.pkl"
        )
        assert os.path.exists(node_label_event_file_path), (
            "node node event (unsorted) does not exist at"
            f"{node_label_event_file_path}"
        )

        node_label_event_sorted_file_path = os_join(
            self.node_label_event_sorted_path, "node_label_event_sorted.pkl"
        )
        if os.path.exists(node_label_event_sorted_file_path):
            if self.verbose:
                print(
                    "sorted node label already exists at: "
                    f"{node_label_event_sorted_file_path}"
                )
        else:
            if not os.path.exists(self.node_label_event_sorted_path):
                os.makedirs(self.node_label_event_sorted_path)

            node_label_event_file: List[Any] = None
            with open(node_label_event_file_path, "rb") as f:
                node_label_event_file = pickle.load(f)

            # sorted_buf = [(u:str, label:int, t:float), ...]
            # sorted_buf = sorted(node_label_event_file, key=lambda x: x[2])
            node_label_event_file.sort(key=lambda x: x[2])
            sorted_buf = node_label_event_file
            with open(node_label_event_sorted_file_path, "wb") as f:
                pickle.dump(sorted_buf, f)
        return None

    def download_parse_sort_data(self):
        """download/parse/sort data for splitting into snapshots."""

        self.download_raw_data()
        self.parse_edge_event()
        self.sort_edge_event()

        self.parse_node_feature_event()
        self.sort_node_feature_event()

        self.parse_node_label_event()
        self.sort_node_label_event()

        return None

    def _generate_graph_event_snapshots(
        self,
        interval: Any,
        base_snapshot_t: Any,
        graph_event_split_path: str,
        edge_event_split_path: str,
        node_feature_event_split_path: str,
        node_label_event_split_path: str,
        graph_snapshot_metadata_path: str,
    ):
        """Use interval to split edge/node-feature/node-label list

        write to local graph snaphots directory.
        """

        total_snapshot = 0
        total_num_nodes = 0
        snapshot_interval = interval
        base_snapshot_t = base_snapshot_t
        edge_event_timestamp_max = 0.0
        edge_event_timestamp_min = 0.0
        node_feat_dim = None
        node_label_num_class = None
        num_edge_events_per_snapshot = []
        num_node_per_snapshot = []
        edge_event_split_path_list = []
        node_feature_event_split_path_list = []
        node_label_snap_file_path_list = []

        node_set = set()

        assert not os.path.exists(
            graph_event_split_path
        ), f"graph event snapshot already exists: {graph_event_split_path}"
        os.makedirs(graph_event_split_path)

        # 1: handle edge struct events

        assert not os.path.exists(
            edge_event_split_path
        ), f"edge event snapshot already exists: {edge_event_split_path}"
        os.makedirs(edge_event_split_path)

        _snapshot_id: int = 0
        next_snapshot_t = base_snapshot_t
        write_out_list = []

        with open(
            os_join(self.edge_event_sorted_path, "data.json"),
            "r",
            encoding="utf-8",
        ) as f:
            for line in f:
                u, v, delta_w, t = json.loads(line.strip())
                edge_event_timestamp_min = min([edge_event_timestamp_min, t])
                edge_event_timestamp_max = max([edge_event_timestamp_max, t])
                node_set.add(u)
                node_set.add(v)

                if float(t) <= next_snapshot_t:
                    write_out_list.append(line.strip())
                else:
                    # increase next_snapshot_t to make t reach it.
                    # making empty snapshots.
                    while float(t) > next_snapshot_t:
                        edge_event_file_path = os_join(
                            edge_event_split_path,
                            f"{self.graph_event_prefix}{_snapshot_id}.json",
                        )
                        edge_event_split_path_list.append(edge_event_file_path)
                        # record current snapshot stats
                        # write to file
                        num_edge_events_per_snapshot.append(
                            len(write_out_list),
                        )
                        num_node_per_snapshot.append(len(node_set))
                        self.__write_json(edge_event_file_path, write_out_list)
                        write_out_list = []

                        _snapshot_id += 1
                        next_snapshot_t = next_snapshot_t + interval

                    write_out_list.append(line.strip())

            # store residual for edge event
            if len(write_out_list) > 0:
                edge_event_file_path = os_join(
                    edge_event_split_path,
                    f"{self.graph_event_prefix}{_snapshot_id}.json",
                )
                edge_event_split_path_list.append(edge_event_file_path)
                self.__write_json(edge_event_file_path, write_out_list)
                num_edge_events_per_snapshot.append(len(write_out_list))
                num_node_per_snapshot.append(len(node_set))
                write_out_list = []

            total_num_nodes = len(node_set)
            total_snapshot = len(num_node_per_snapshot)

        # 2: handle node feature events
        assert not os.path.exists(node_feature_event_split_path), (
            "graph node feature snapshot already exists:            "
            f" {node_feature_event_split_path}"
        )
        os.makedirs(node_feature_event_split_path)

        num_node_feature_events_per_snapshot = []
        node_feature_sorted_file_path = os_join(
            self.node_feat_event_sorted_path, "node_feature_event_sorted.pkl"
        )
        if not os.path.exists(node_feature_sorted_file_path):
            # no exist node feature pickle file
            # generate empty snap file
            self.__init_empty_node_feat_snapshots(
                node_feature_event_split_path,
                total_snapshot,
                num_node_feature_events_per_snapshot,
                node_feature_event_split_path_list,
            )
        else:
            # init all empty snapshots
            self.__init_empty_node_feat_snapshots(
                node_feature_event_split_path,
                total_snapshot,
                num_node_feature_events_per_snapshot,
                node_feature_event_split_path_list,
            )
            with open(node_feature_sorted_file_path, "rb") as f:
                node_feat_sorted_list = pickle.load(f)
            _snapshot_id: int = 0
            node_feat_snap_file_path = os_join(
                node_feature_event_split_path,
                f"{self.graph_event_prefix}{_snapshot_id}.pkl",
            )
            next_snapshot_t = base_snapshot_t
            write_out_list = []
            for i in range(len(node_feat_sorted_list)):
                u, feat_v, t = node_feat_sorted_list[i]
                if node_feat_dim is None:
                    node_feat_dim = feat_v.shape[0]

                if float(t) <= next_snapshot_t:
                    write_out_list.append(node_feat_sorted_list[i])
                else:
                    # increase next_snapshot_t to make t reach it.
                    # making empty snapshots.
                    while (
                        float(t) > next_snapshot_t
                        and _snapshot_id <= total_snapshot - 1
                    ):
                        node_feat_snap_file_path = os_join(
                            node_feature_event_split_path,
                            f"{self.graph_event_prefix}{_snapshot_id}.pkl",
                        )
                        # record current snapshot stats
                        # write to file
                        num_node_feature_events_per_snapshot[
                            _snapshot_id
                        ] = len(write_out_list)

                        with open(node_feat_snap_file_path, "wb") as f:
                            pickle.dump(write_out_list, f)
                        write_out_list = []

                        _snapshot_id += 1
                        next_snapshot_t = next_snapshot_t + interval

                    if _snapshot_id > total_snapshot - 1:
                        print(
                            "node feature events have more snapshots than"
                            "edge events; ignore those events"
                        )
                        break
                    write_out_list.append(node_feat_sorted_list[i])

            # store residual for nod event
            if len(write_out_list) > 0:
                num_node_feature_events_per_snapshot[_snapshot_id] = len(
                    write_out_list
                )
                node_feat_snap_file_path = os_join(
                    node_feature_event_split_path,
                    f"{self.graph_event_prefix}{_snapshot_id}.pkl",
                )
                with open(node_feat_snap_file_path, "wb") as f:
                    pickle.dump(write_out_list, f)
                write_out_list = []

        # 3: handle node label events
        assert not os.path.exists(node_label_event_split_path), (
            "graph node label snapshot already exists:            "
            f" {node_label_event_split_path}"
        )
        os.makedirs(node_label_event_split_path)

        num_node_label_events_per_snapshot = []
        node_label_event_sorted_file_path = os_join(
            self.node_label_event_sorted_path,
            "node_label_event_sorted.pkl",
        )

        if not os.path.exists(node_label_event_sorted_file_path):
            # no exist node feature pickle file
            # generate empty snap file
            self.__init_empty_node_lb_snapshots(
                node_label_event_split_path,
                total_snapshot,
                num_node_label_events_per_snapshot,
                node_label_snap_file_path_list,
            )
        else:
            self.__init_empty_node_lb_snapshots(
                node_label_event_split_path,
                total_snapshot,
                num_node_label_events_per_snapshot,
                node_label_snap_file_path_list,
            )
            with open(node_label_event_sorted_file_path, "rb") as f:
                node_label_event_sorted_list = pickle.load(f)
            _snapshot_id: int = 0
            node_label_snap_file_path = os_join(
                node_label_event_split_path,
                f"{self.graph_event_prefix}{_snapshot_id}.pkl",
            )
            next_snapshot_t = base_snapshot_t
            write_out_list = []
            for i in range(len(node_label_event_sorted_list)):
                u, label, t = node_label_event_sorted_list[i]
                if node_label_num_class is None:
                    node_label_num_class = label.shape[0]

                if float(t) <= next_snapshot_t:
                    write_out_list.append(node_label_event_sorted_list[i])
                else:
                    # increase next_snapshot_t to make t reach it.
                    # making empty snapshots.
                    while (
                        float(t) > next_snapshot_t
                        and _snapshot_id <= total_snapshot - 1
                    ):
                        node_label_snap_file_path = os_join(
                            node_label_event_split_path,
                            f"{self.graph_event_prefix}{_snapshot_id}.pkl",
                        )
                        # record current snapshot stats
                        # write to file
                        num_node_label_events_per_snapshot[_snapshot_id] = len(
                            write_out_list
                        )

                        with open(node_label_snap_file_path, "wb") as f:
                            pickle.dump(write_out_list, f)
                        write_out_list = []

                        _snapshot_id += 1
                        next_snapshot_t = next_snapshot_t + interval

                    if _snapshot_id > total_snapshot - 1:
                        print(
                            "node label events have more snapshots than"
                            "edge events; ignore those events"
                        )
                        break
                    write_out_list.append(node_label_event_sorted_list[i])

            # store residual for nod event
            if len(write_out_list) > 0:
                num_node_label_events_per_snapshot[_snapshot_id] = len(
                    write_out_list
                )
                node_label_snap_file_path = os_join(
                    node_label_event_split_path,
                    f"{self.graph_event_prefix}{_snapshot_id}.pkl",
                )
                with open(node_label_snap_file_path, "wb") as f:
                    pickle.dump(write_out_list, f)
                write_out_list = []

        assert node_label_num_class is not None, (
            "node_label_num_class should not be None, current it is"
            f"{node_label_num_class}"
        )
        assert (
            node_feat_dim is not None
        ), f"node_feat_dim should not be None, current it is{node_feat_dim}"

        graph_metadata = DynGraphMetadata(
            total_snapshot,
            total_num_nodes,
            snapshot_interval,
            base_snapshot_t,
            edge_event_timestamp_max,
            edge_event_timestamp_min,
            node_feat_dim,
            node_label_num_class,
            num_edge_events_per_snapshot,
            num_node_per_snapshot,
            num_node_feature_events_per_snapshot,
            num_node_label_events_per_snapshot,
            edge_event_split_path_list,
            node_feature_event_split_path_list,
            node_label_snap_file_path_list,
        )

        if self.verbose:
            print("graph metadata:")
            print(graph_metadata)

        # write out graph_metadata
        with open(graph_snapshot_metadata_path, "w", encoding="utf-8") as f:
            f.write(graph_metadata.to_json())

        return graph_metadata

    def __init_empty_node_lb_snapshots(
        self,
        node_label_event_split_path,
        total_snapshot,
        num_node_label_events_per_snapshot,
        node_label_snap_file_path_list,
    ):
        for _snapshot_id in range(total_snapshot):
            node_label_snap_file_path = os_join(
                node_label_event_split_path,
                f"{self.graph_event_prefix}{_snapshot_id}.pkl",
            )
            node_label_snap_file_path_list.append(node_label_snap_file_path)

            with open(node_label_snap_file_path, "wb") as f:
                pickle.dump([], f)
            num_node_label_events_per_snapshot.append(0)

    def __init_empty_node_feat_snapshots(
        self,
        node_feature_event_split_path,
        total_snapshot,
        num_node_feature_events_per_snapshot,
        node_feature_event_split_path_list,
    ):
        for _snapshot_id in range(total_snapshot):
            node_feat_snap_file_path = os_join(
                node_feature_event_split_path,
                f"{self.graph_event_prefix}{_snapshot_id}.pkl",
            )
            node_feature_event_split_path_list.append(node_feat_snap_file_path)
            with open(node_feat_snap_file_path, "wb") as f:
                pickle.dump([], f)
            num_node_feature_events_per_snapshot.append(0)

    def get_graph_event_snapshots_from_sorted_events(
        self, interval: Any, base_snapshot_t: Any
    ):
        """get the graph events (edge/node-feature/node-labels) from
        generated snapshot data."

        Args:
            interval (Any): the time interval for one snapshot
            base_snapshot_t (Any): the max timestamp for the first
                snapshot

        Returns:
            list (DynGraphMetadata): the metadata of generated graph
                snapshots
        """

        graph_event_split_path: str = os_join(
            self.local_graph_dataset_dir_abs_path,
            f"graph_snapshot_base_{base_snapshot_t}_interval_{interval}",
        )
        edge_event_split_path: str = os_join(
            graph_event_split_path,
            "edge_event_snapshots",
        )
        node_feature_event_split_path: str = os_join(
            graph_event_split_path,
            "node_feature_event_snapshots",
        )
        node_label_event_split_path: str = os_join(
            graph_event_split_path,
            "node_label_event_snapshots",
        )
        graph_snapshot_metadata_path: str = os_join(
            graph_event_split_path,
            "graph_snapshot_metadata.json",
        )

        if (
            os.path.exists(graph_event_split_path)
            and os.path.exists(edge_event_split_path)
            and os.path.exists(node_feature_event_split_path)
            and os.path.exists(node_label_event_split_path)
        ):
            _graph_snaphot_metadata_path: str = os_join(
                graph_event_split_path, "graph_snapshot_metadata.json"
            )
            graph_metadata: DynGraphMetadata = None
            with open(
                _graph_snaphot_metadata_path, "r", encoding="utf-8"
            ) as f:
                graph_metadata = DynGraphMetadata.from_json(
                    f.readlines()[0].strip()
                )

            if self.verbose:
                print("===\tStart: graph snapshot metadata:\t ===")
                print(graph_metadata)
                print("===\tEnd: graph snapshot metadata:\t ===")

        else:
            graph_metadata = self._generate_graph_event_snapshots(
                interval,
                base_snapshot_t,
                graph_event_split_path,
                edge_event_split_path,
                node_feature_event_split_path,
                node_label_event_split_path,
                graph_snapshot_metadata_path,
            )

        return graph_metadata

    def get_train_val_test_node_ids(
        self,
        interval: Any,
        base_snapshot_t: Any,
    ):
        """Get the train/val/test labels for each snapshot.
        Iterate graphmeta snapshot data, recording which nodes appeared
        at each snapshot; Then pick nodes into the train/validate/test
        disjoint set. Make sure the nodes have label.

        """

        raise NotImplementedError

    @staticmethod
    def __write_json(path, writable_list):
        with open(path, "w", encoding="utf-8") as f:
            for _ in writable_list:
                f.write(_)
                f.write("\n")

    @staticmethod
    def load_json_file(path):
        if path is None:
            data = None
        else:
            data = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    u, v, delta_w, t = json.loads(line.strip())
                    data.append(
                        (
                            str(u),
                            str(v),
                            float(delta_w),
                            float(t),
                        )
                    )
        return data

    @staticmethod
    def load_pkl_file(path):
        if path is None:
            data = None
        else:
            with open(path, "rb") as f:
                data = pickle.load(f)
        return data
