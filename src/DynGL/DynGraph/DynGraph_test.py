import unittest
import numpy as np
from DynGraph import DynGraph


class test_dyngl(unittest.TestCase):
    def test_graph_struct_change(self):
        edge_list = [
            [
                (0, 1, 1.0),
            ],
            [
                (1, 2, 1.0),
            ],
            [
                (2, 1, 3.5),
            ],
        ]

        max_node_num = 3
        node_feat_dim = 10
        edge_feat_dim = 10
        dyn_graph = DynGraph(
            max_node_num=max_node_num,
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
        )

        for timestamp, edge_e in enumerate(edge_list):
            dyn_graph.update_graph(
                update_timestamp=timestamp,
                edge_struct_change=edge_e,
            )
            print(f"-----after {timestamp} -----")
            print(dyn_graph.csr_graph.todense())
            print(dyn_graph.degree_in)
            print(
                np.squeeze(
                    np.asarray(np.sum(dyn_graph.csr_graph.todense(), axis=0))
                )
            )

            print(dyn_graph.degree_out)
            print(
                np.squeeze(
                    np.asarray(np.sum(dyn_graph.csr_graph.todense(), axis=1))
                )
            )

    def test_graph_struct_and_edge_feat_change(self):
        edge_list = [
            [
                (0, 1, 1.0),
                (2, 1, 1.5),
            ],
            [
                (1, 2, 1.0),
            ],
            [
                (2, 1, 3.5),
            ],
        ]

        edge_feat_list = [
            [],
            [
                (0, 1, np.array([0.1, 0.1])),
                (1, 2, np.array([0.2, 0.2])),
                (2, 1, np.array([0.15, 0.15])),
                # (1, 0, np.array([0.1, 0.1])),
            ],
            [
                (2, 1, np.array([0.3, 0.3])),
                (0, 1, np.array([0.9, 0.9])),
            ],
        ]

        max_node_num = 3
        node_feat_dim = 2
        edge_feat_dim = 2
        dyn_graph = DynGraph(
            max_node_num=max_node_num,
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
        )

        for timestamp, (edge_struct_e, edge_feat_e) in enumerate(
            zip(edge_list, edge_feat_list)
        ):
            dyn_graph.update_graph(
                update_timestamp=timestamp,
                edge_struct_change=edge_struct_e,
                edge_features_override=edge_feat_e,
            )
            print(f"-----after {timestamp} -----")
            print("graph:\n", dyn_graph.csr_graph.todense())
            degree_in_csr = np.squeeze(
                np.asarray(np.sum(dyn_graph.csr_graph.todense(), axis=0))
            )
            print(
                "in-degree:\n",
                dyn_graph.degree_in,
                np.array_equal(dyn_graph.degree_in, degree_in_csr),
            )

            print("out-degree:\n", dyn_graph.degree_out)
            degree_out_csr = np.squeeze(
                np.asarray(np.sum(dyn_graph.csr_graph.todense(), axis=1))
            )
            print(
                "in-degree:\n",
                dyn_graph.degree_out,
                np.array_equal(dyn_graph.degree_out, degree_out_csr),
            )

            print("edge feature:\n", dyn_graph.edge_feature)

    def test_graph_struct_and_edge_node_feat_change(self):
        edge_list = [
            [
                (0, 1, 1.0),
            ],
            [
                (1, 2, 1.0),
            ],
            [
                (2, 1, 3.5),
            ],
        ]

        edge_feat_list = [
            [],
            [
                (0, 1, np.array([0.1, 0.1])),
                (1, 2, np.array([0.2, 0.2])),
                # (1, 0, np.array([0.1, 0.1])),
            ],
            [
                (2, 1, np.array([0.3, 0.3])),
                (0, 1, np.array([0.9, 0.9])),
            ],
        ]

        node_feat_list = [
            [
                (0, np.array([0.1, 0.1])),
                # (2, np.array([0.1, 0.1])),
            ],
            [
                (0, np.array([0.25, 0.25])),
                (1, np.array([0.2, 0.2])),
            ],
            [
                (0, np.array([0.6, 0.6])),
                (1, np.array([0.7, 0.7])),
                (2, np.array([0.8, 0.8])),
            ],
        ]

        max_node_num = 3
        node_feat_dim = 2
        edge_feat_dim = 2
        dyn_graph = DynGraph(
            max_node_num=max_node_num,
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
        )

        for timestamp, (edge_struct_e, edge_feat_e, node_feat_e) in enumerate(
            zip(edge_list, edge_feat_list, node_feat_list)
        ):
            dyn_graph.update_graph(
                update_timestamp=timestamp,
                edge_struct_change=edge_struct_e,
                edge_features_override=edge_feat_e,
                node_feature_override=node_feat_e,
            )
            print(f"-----after {timestamp} -----")
            print("graph:\n", dyn_graph.csr_graph.todense())
            degree_in_csr = np.squeeze(
                np.asarray(np.sum(dyn_graph.csr_graph.todense(), axis=0))
            )
            print(
                "in-degree:\n",
                dyn_graph.degree_in,
                np.array_equal(dyn_graph.degree_in, degree_in_csr),
            )

            print("out-degree:\n", dyn_graph.degree_out)
            degree_out_csr = np.squeeze(
                np.asarray(np.sum(dyn_graph.csr_graph.todense(), axis=1))
            )
            print(
                "in-degree:\n",
                dyn_graph.degree_out,
                np.array_equal(dyn_graph.degree_out, degree_out_csr),
            )

            print("edge feature:\n", dyn_graph.edge_feature)
            print("node feature:\n", dyn_graph.node_feature)

    def test_graph_struct_and_edge_node_feat_node_label_change(self):
        edge_list = [
            [
                (0, 1, 1.0),
            ],
            [
                (1, 2, 1.0),
            ],
            [
                (2, 1, 3.5),
            ],
        ]

        edge_feat_list = [
            [],
            [
                (0, 1, np.array([0.1, 0.1])),
                (1, 2, np.array([0.2, 0.2])),
                # (1, 0, np.array([0.1, 0.1])),
            ],
            [
                (2, 1, np.array([0.3, 0.3])),
                (0, 1, np.array([0.9, 0.9])),
            ],
        ]

        node_feat_list = [
            [
                (0, np.array([0.1, 0.1])),
                # (2, np.array([0.1, 0.1])),
            ],
            [
                (0, np.array([0.25, 0.25])),
                (1, np.array([0.2, 0.2])),
            ],
            [
                (0, np.array([0.6, 0.6])),
                (1, np.array([0.7, 0.7])),
                (2, np.array([0.8, 0.8])),
            ],
        ]

        node_label_list = [
            [
                # (0, 0),
                # (1, 0),
            ],
            [
                (1, 1),
                (2, 1),
            ],
            [
                (0, 2),
                (1, 2),
                (2, 2),
            ],
        ]

        max_node_num = 3
        node_feat_dim = 2
        edge_feat_dim = 2
        dyn_graph = DynGraph(
            max_node_num=max_node_num,
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
        )

        for timestamp, (
            edge_struct_e,
            edge_feat_e,
            node_feat_e,
            node_lb_e,
        ) in enumerate(
            zip(edge_list, edge_feat_list, node_feat_list, node_label_list)
        ):
            dyn_graph.update_graph(
                update_timestamp=timestamp,
                edge_struct_change=edge_struct_e,
                edge_features_override=edge_feat_e,
                node_feature_override=node_feat_e,
                node_label_override=node_lb_e,
            )
            print(f"-----after {timestamp} -----")
            print("graph:\n", dyn_graph.csr_graph.todense())
            degree_in_csr = np.squeeze(
                np.asarray(np.sum(dyn_graph.csr_graph.todense(), axis=0))
            )
            print(
                "in-degree:\n",
                dyn_graph.degree_in,
                np.array_equal(dyn_graph.degree_in, degree_in_csr),
            )

            print("out-degree:\n", dyn_graph.degree_out)
            degree_out_csr = np.squeeze(
                np.asarray(np.sum(dyn_graph.csr_graph.todense(), axis=1))
            )
            print(
                "in-degree:\n",
                dyn_graph.degree_out,
                np.array_equal(dyn_graph.degree_out, degree_out_csr),
            )

            print("edge feature:\t\n", dyn_graph.edge_feature)
            print("node feature:\t\n", dyn_graph.node_feature)
            print("node label:\t\n", dyn_graph.node_labels)


if __name__ == "__main__":
    unittest.main()
