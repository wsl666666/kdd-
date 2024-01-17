from DynGraphLoader import DynGraphReader
import os
from os.path import join as os_join
import json
from overrides import override
import numpy as np
import pickle


class DynGraphReaderKarateTest(DynGraphReader):
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


if __name__ == "__main__":
    karate_dataset_reader = DynGraphReaderKarateTest(
        graph_dataset_name="karate-test",
        local_dataset_dir_abs_path="/home/xingzguo/projects_data/DynMixer/",
        verbose=True,
    )

    karate_dataset_reader.download_parse_sort_data()
    karate_dataset_reader.get_graph_event_snapshots_from_sorted_events(
        interval=4.0,
        base_snapshot_t=40.0,
    )

    karate_dataset_reader.get_graph_event_snapshots_from_sorted_events(
        interval=4.0,
        base_snapshot_t=50.0,
    )

    karate_dataset_reader.get_graph_event_snapshots_from_sorted_events(
        interval=4.0,
        base_snapshot_t=60.0,
    )
