from typing import List, Dict, Any, Tuple, Callable
from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix


@dataclass
class EdgeStructChangeEvent:
    """class for edge events."""

    src_node_raw: Any
    src_node_id: int
    tgt_node_raw: Any
    tgt_node_id: int
    # the edge weight change
    delta_weight: float


class DynGraph:
    """Snapshots-based dynamic graphs."""

    def __init__(
        self,
        max_node_num: int,
        node_feat_dim: int,
        edge_feat_dim: int,
        node_label_none: bool = False,
        node_label_num_class: int = 2,
        edge_label_none: int = -1,
        edge_label_num_class: int = 2,
    ):
        """maintain graphs information for dynamic settings

        # TODO: handle #max node is unknown.
        # NOTE: node raw description must be hashable.

        -- maintain the mapping of node_raw_id to node_id
        -- maintain sparse matrics (csr, coo, ...) for edge events
        -- maintain the degree vector
        -- maintain node features for node event
        -- maintain node label for node event.

        """
        self.max_node_num = max_node_num
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.node_label_none = node_label_none
        self.node_label_num_class = node_label_num_class
        self.edge_label_none = edge_label_none
        self.edge_label_num_class = edge_label_num_class

        assert (
            not self.node_label_none
        ), f"node_label_none ({node_label_none}) should be false"
        assert (
            self.node_label_num_class >= 2
        ), f"node_label_num_class ({self.node_label_num_class}) should >=2"

        assert (
            self.edge_label_none < 0
        ), f"edge_label_none ({edge_label_none}) should be negative"
        assert (
            self.edge_label_num_class >= 2
        ), f"edge_label_num_class ({self.edge_label_num_class}) should >=2"
        # map for node raw desc <--> node id
        self.node_raw_id_map: Dict[Any, int] = {}
        self.node_id_raw_map: Dict[int, Any] = {}

        self.latest_update_timestamp: float = None

        # graph structure
        self.csr_graph = csr_matrix(
            (self.max_node_num, self.max_node_num),
            dtype=float,
        )

        # node info
        self.degree_in = np.zeros(self.max_node_num, dtype=float)
        self.degree_out = np.zeros(self.max_node_num, dtype=float)
        # FIXME: add feature dynamically to save run-time memory?
        # Original implementation supports it.
        # But it cause problem in sparse matrix multiplication.
        # self.node_feature = np.zeros((0, self.node_feat_dim), dtype=float)
        self.node_feature = np.zeros(
            (self.max_node_num, self.node_feat_dim),
            dtype=float,
        )
        self.node_feature.fill(np.nan)
        self.node_labels = np.ones(
            (self.max_node_num, self.node_label_num_class), dtype=bool
        )
        self.node_labels.fill(False)

        # edge info (if any)
        self.edge_id: Dict[Tuple[int, int], int] = {}  # node id pair
        # convert edge id to fetch the idx of edge feature
        # note that not every edge has feature
        self.edge_id_to_edge_feat_idx: Dict[int, int] = {}
        self.edge_feature = np.zeros((0, edge_feat_dim), dtype=float)
        self.edge_label = np.array([], dtype=int)
        self.edge_label.fill(self.edge_label_none)

    def get_node_id(
        self,
        node_raw: Any,
        auto_create: bool = True,
    ) -> Tuple[int, bool]:
        """get node id from node raw desc"""
        is_new_node = None

        if node_raw in self.node_raw_id_map:
            node_id = self.node_raw_id_map[node_raw]
            is_new_node = False
            return (node_id, is_new_node)
        if auto_create:
            node_id = self._init_new_node(node_raw)
            is_new_node = True
            return (node_id, is_new_node)
        raise KeyError(
            f"KeyError: {node_raw} not in node_raw map. Is it a unseen node?"
        )

    def _expand_node_feat(self, node_id: int):
        """expand node feature matrix with np.nan. Assume that every
        node has node feature that may change over time.

        """
        if self.node_feature.shape[0] == self.max_node_num:
            return None
        else:
            _feat = np.zeros((1, self.node_feat_dim), dtype=float)
            _feat.fill(np.nan)
            self.node_feature = np.vstack((self.node_feature, _feat))
            assert self.node_feature.shape[0] - 1 == node_id, (
                f"new node_id {node_id}!= node_feature.shape[0]-1"
                f"({self.node_feature.shape[0]}-1)"
            )
            return None

    def _expand_node_in_degree(self, node_id: int):
        """expand node in-degree vector with np.nan"""
        self.degree_in = self.__array_expand_helper(
            self.degree_in,
            node_id,
            self.max_node_num,
            fill_in_val=0.0,
        )

    def _expand_node_out_degree(self, node_id: int):
        """expand node out-degree vector with np.nan"""
        self.degree_out = self.__array_expand_helper(
            self.degree_out,
            node_id,
            self.max_node_num,
            fill_in_val=0.0,
        )

    def _expand_node_label(self, node_id: int):
        """expand node label vector with np.nan"""
        # self.node_labels = self.__array_expand_helper(
        #     self.node_labels, node_id, self.max_node_num
        # )
        # we init node labels.
        pass

    def _init_new_node(self, node_raw: Any) -> int:
        """encounter a new node, assign and return the new ID;
        initialize the corresponding matrix/vector

        """
        node_id = len(self.node_raw_id_map)
        self.node_raw_id_map[node_raw] = node_id
        self.node_id_raw_map[node_id] = node_raw

        # add new dim to node feature
        self._expand_node_feat(node_id)
        # add new dim to node degree
        self._expand_node_in_degree(node_id)
        self._expand_node_out_degree(node_id)
        # add new dim to node label
        self._expand_node_label(node_id)
        return node_id

    def update_graph(
        self,
        update_timestamp: float,
        edge_struct_change: List[Tuple[Any, Any, float]] = None,
        edge_features_override: List[Tuple[Any, Any, np.ndarray]] = None,
        edge_label_override: List[Tuple[Any, Any, int]] = None,
        node_feature_override: List[Tuple[Any, np.ndarray]] = None,
        node_label_override: List[Tuple[Any, int]] = None,
        callback_handle_func_single_edge_struct_event_after: Callable = None,
        callback_handle_func_single_edge_struct_event_before: Callable = None,
        callback_handle_func_all_edge_struct_event_after: Callable = None,
        *args,
        **kwargs,
    ):
        """update graphs, including edge structure/weight, edge features
        , edge labels, node features, node labels, and edge

        Args:
            update_timestamp (float): the timestamp for this batch
            edge_struct_change (List[Tuple[Any, Any, float]], optional):
                the edge link and weight changes. Defaults to None.
            edge_features_override (List[Tuple[Any, Any, np.ndarray]],
                optional): the edge feature changes override . Defaults
                to None.
            edge_label_override (List[Tuple[Any, Any, int]], optional):
                the edge label changes override. Defaults to None.
            node_feature_override (List[Tuple[Any, np.ndarray]],
                optional): the node feature changes override.
                Defaults to None.
            node_label_override (List[Tuple[Any, int]], optional):
                the node label changes override. Defaults to None.
            callback_handle_func_single_edge_struct_event_after
                (Callable): the callback function after one edge
                event being recorded.
            callback_handle_func_single_edge_struct_event_before
                (Callable):the callback function before one edge
                event being recorded.
        """
        # update graph with input changes
        self.latest_update_timestamp = update_timestamp

        # Step 1: update edge structure
        if edge_struct_change is not None:
            # update edge structure
            self._update_edges_structure(
                update_timestamp,
                edge_struct_change,
                callback_handle_func_single_edge_struct_event_after,
                callback_handle_func_single_edge_struct_event_before,
                callback_handle_func_all_edge_struct_event_after,
            )

        if edge_features_override is not None:
            self._update_edges_feature(
                update_timestamp,
                edge_features_override,
            )

        if edge_label_override is not None:
            self._update_edges_label(update_timestamp, edge_label_override)

        if node_feature_override is not None:
            self._update_node_feature(update_timestamp, node_feature_override)

        if node_label_override is not None:
            self._update_node_label(update_timestamp, node_label_override)

    def _update_edges_structure(
        self,
        update_timestamp: float,
        edge_struct_events: List[Tuple[Any, Any, float]],
        callback_handle_func_single_edge_struct_event_after: Callable = None,
        callback_handle_func_single_edge_struct_event_before: Callable = None,
        callback_handle_func_all_edge_struct_event_after: Callable = None,
        *args,
        **kwargs,
    ):
        """update graph structure from a batch of edge structure changes
        . The

        Args:
            update_timestamp (float): the timestamp for this batch
            edge_struct_events (List[Tuple[Any, Any, float]]):
                the list edge structure changes e.g,
                (
                    src_node_raw_tag,
                    tgt_node_raw_tag,
                    delta_edge_weight
                )
        """
        update_kwargs = {}

        # modify the edge weight in sparase matrix and degree vector
        (
            struct_events_consolidate,
            update_kwargs,
        ) = self.consolidate_edge_structure_events(
            edge_struct_events,
            update_kwargs,
        )
        total_events_consolidate = len(struct_events_consolidate.keys())

        (
            cache_before_update,
            update_kwargs,
        ) = self.handle_func_all_edge_struct_event_before(
            total_events_consolidate, update_kwargs
        )

        # iterate all consilidated edge structure events to update:
        # -- CSR sparase matrix
        # -- in/out-degree vectors
        # -- execute external callback function. e.g., ppr adjustment
        # per edge
        # This for loop record any intermediate changes in delta state.
        # intermiediate changes are stored in cache_before_update
        # Finally apply delta to original csr mat and in/out-degree
        # vectors

        total_iter = len(struct_events_consolidate)
        for e_iter, (uv_pair, e_edge) in enumerate(
            struct_events_consolidate.items(),
        ):
            src = e_edge.src_node_id
            tgt = e_edge.tgt_node_id
            delta_w_consolidate = e_edge.delta_weight

            # add interactive
            if e_iter % 1000 == 0:
                print(f"#Edge event updated: {e_iter} / {total_iter}", end="\r")

            update_kwargs = self.handle_func_single_edge_struct_event_before(
                src,
                tgt,
                delta_w_consolidate,
                e_iter,
                cache_before_update,
                update_kwargs,
                callback_handle_func_single_edge_struct_event_before,
            )
            # add edge to graph with delta-weight
            update_kwargs = self.handle_record_edge_structure_changes(
                src,
                tgt,
                delta_w_consolidate,
                e_iter,
                cache_before_update,
                update_kwargs,
            )

            # a = cache_before_update["_out_degree"]
            # b = cache_before_update["_delta_out_degree"]

            update_kwargs = self.handle_func_single_edge_struct_event_after(
                src,
                tgt,
                delta_w_consolidate,
                e_iter,
                cache_before_update,
                update_kwargs,
                callback_handle_func_single_edge_struct_event_after,
            )
        print(f"#Edge event updated: {e_iter} / {total_iter}")
        # apply consolidate edge struct delta changes and degree vects
        update_kwargs = self.handle_func_all_edge_struct_event_after(
            cache_before_update,
            update_kwargs,
            callback_handle_func_all_edge_struct_event_after,
        )

    def consolidate_edge_structure_events(
        self,
        edge_struct_events: List[Tuple[Any, Any, float]],
        update_kwargs: Dict = None,
    ) -> Dict[Tuple[int, int], EdgeStructChangeEvent]:
        """consolidate edge events by merging (u, v, delta_w)
        based on (u,v) pairs. Unseen nodes will be initialized.

        Args:
            edge_struct_events (List[Tuple[Any, Any, float]]): raw edge
            struct events
            update_kwargs (Dict): additional update kwargs for handler
                functions

        Returns:
            Dict[Tuple[int, int], EdgeStructChangeEvent]:
            the consolidated edge change events
        """
        structure_events_dict: Dict[
            Tuple[int, int],
            EdgeStructChangeEvent,
        ] = {}

        for _d in edge_struct_events:
            src_node_raw = _d[0]
            src_node_id, is_src_new_node = self.get_node_id(src_node_raw)

            tgt_node_raw = _d[1]
            tgt_node_id, is_tgt_new_node = self.get_node_id(tgt_node_raw)

            delta_weight = _d[2]

            if src_node_id == tgt_node_id:
                print(f"self-lopp has been removed {src_node_raw}")
                continue

            assert isinstance(
                delta_weight, float
            ), f"delta_weight must be a float, but it is type{delta_weight}"

            # update edge_id
            _event_key = (src_node_id, tgt_node_id)
            self.update_edge_id(_event_key)

            # consolidate edge weight changes.
            if _event_key not in structure_events_dict:
                structure_events_dict[_event_key] = EdgeStructChangeEvent(
                    src_node_raw,
                    src_node_id,
                    tgt_node_raw,
                    tgt_node_id,
                    delta_weight,
                )
            else:
                # Accumulated weight changes for (src, tgt) pair
                # in this snapshot
                structure_events_dict[_event_key].delta_weight += delta_weight
        return structure_events_dict, update_kwargs

    def handle_record_edge_structure_changes(
        self,
        src: int,
        tgt: int,
        delta_w_consolidate: float,
        e_iter: int,
        cache_before_update: Dict,
        update_kwargs: Dict = None,
    ):
        """record Delta (changes) in csr matrix and degree vectors"""
        # record delta in csr matrix
        cache_before_update["_new_src"][e_iter] = src
        cache_before_update["_new_tgt"][e_iter] = tgt
        cache_before_update["_delta_w"][e_iter] = delta_w_consolidate

        # record delta in/out-degree vectors
        cache_before_update["_delta_in_degree"][tgt] += delta_w_consolidate
        cache_before_update["_delta_out_degree"][src] += delta_w_consolidate

        # record intermedaite in/out-degree vector
        cache_before_update["_crt_intermediate_in_degree"][tgt] += delta_w_consolidate
        cache_before_update["_crt_intermediate_out_degree"][src] += delta_w_consolidate

        return update_kwargs

    def handle_func_single_edge_struct_event_before(
        self,
        src: int,
        tgt: int,
        delta_w_consolidate: float,
        e_iter: int,
        cache_before_update: Dict,
        update_kwargs: Dict = None,
        callback_handle_func_single_edge_struct_event_before: Callable = None,
    ):
        """handler before a single consolidated edge got recorded"""
        if callback_handle_func_single_edge_struct_event_before is not None:
            callback_handle_func_single_edge_struct_event_before(
                src,
                tgt,
                delta_w_consolidate,
                e_iter,
                cache_before_update,
                update_kwargs,
            )
        return update_kwargs

    def handle_func_single_edge_struct_event_after(
        self,
        src: int,
        tgt: int,
        delta_w_consolidate: float,
        e_iter: int,
        cache_before_update: Dict,
        update_kwargs: Dict = None,
        callback_handle_func_single_edge_struct_event_after: Callable = None,
    ):
        """handler after a single consolidated edge got recorded"""
        if callback_handle_func_single_edge_struct_event_after is not None:
            callback_handle_func_single_edge_struct_event_after(
                src,
                tgt,
                delta_w_consolidate,
                e_iter,
                cache_before_update,
                update_kwargs,
            )
        return update_kwargs

    def handle_func_all_edge_struct_event_before(
        self,
        total_events_consolidate: int,
        update_kwargs: Dict = None,
    ):
        """handler function being called before iterating next batch of
        edge structure events. For example: it initializes the delta
        vectors and delta CSR matrix, which record the structure delta/
        diffs/changes, s.t. prev_status + diff = next_status

        Args:
            total_events_consolidate (int): the total number of
                consolidated edge structure events,

        Returns:
            Dict: the cached status before iterating new edge strcuture events.
            Dict: additional update kwargs for handler functions
        """
        # E.g: prev_node_degree_vec, prev_graph, ...
        cache_before_update = {}
        cache_before_update["_new_src"] = np.zeros(
            total_events_consolidate,
            dtype=int,
        )
        cache_before_update["_new_tgt"] = np.zeros(
            total_events_consolidate,
            dtype=int,
        )
        cache_before_update["_delta_w"] = np.zeros(
            total_events_consolidate, dtype=float
        )
        cache_before_update["_delta_in_degree"] = np.zeros_like(
            self.degree_in,
        )
        cache_before_update["_delta_out_degree"] = np.zeros_like(
            self.degree_out,
        )
        cache_before_update["_in_degree"] = self.degree_in
        cache_before_update["_out_degree"] = self.degree_out

        cache_before_update["_crt_intermediate_in_degree"] = np.copy(self.degree_in)
        cache_before_update["_crt_intermediate_out_degree"] = np.copy(self.degree_out)

        return cache_before_update, update_kwargs

    def handle_func_all_edge_struct_event_after(
        self,
        cache_before_update: Dict,
        update_kwargs: Dict,
        callback_handle_func_all_edge_struct_event_after: Callable = None,
    ) -> None:
        """Handler function being called after iterating all edge
        structure events.
        In-place updates:
        - 1: it applies delta-csr to update csr graph.
        - 2: it applies delta-in/out-degree to update degree vectors

        """
        # 1: Update csr_graph
        delta_csr_mat = csr_matrix(
            (
                cache_before_update["_delta_w"],
                (
                    cache_before_update["_new_src"],
                    cache_before_update["_new_tgt"],
                ),
            ),
            shape=(self.max_node_num, self.max_node_num),
        )
        # NOTE: running time bottleneck
        self.csr_graph += delta_csr_mat

        # 2: Update degree vectors
        self.degree_in += cache_before_update["_delta_in_degree"]
        self.degree_out += cache_before_update["_delta_out_degree"]

        if callback_handle_func_all_edge_struct_event_after is not None:
            callback_handle_func_all_edge_struct_event_after(
                cache_before_update,
                update_kwargs,
            )
        return update_kwargs

    def update_edge_id(self, edge_key: Tuple[int, int]):
        """update edge_id with node-id pairs"""
        if edge_key not in self.edge_id:
            self.edge_id[edge_key] = len(self.edge_id.keys())

    def get_edge_id(self, edge_key: Tuple[int, int]) -> int:
        """get edge id given pair of node ids

        Args:
            edge_key (Tuple[int, int]): the input node ids

        Return:
            (int): edge_id
        """
        assert edge_key in self.edge_id, f"edge_key {edge_key}is not in self.edge_id"

        return self.edge_id[edge_key]

    def map_edge_id_to_edge_mat_idx(
        self,
        edge_id: int,
        auto_create: bool = False,
    ):
        """convert edge id to the index of edge feature.
        Note that not every edge has edge feature.
        If auto_create is True, edge feature will be expanded.

        return
            int: the edge_id's corresponding index in feature matrix

        raise:
            KeyError: if there is no corresponding index for edge
                feature, and auto-create is False.
        """
        if edge_id in self.edge_id_to_edge_feat_idx:
            return self.edge_id_to_edge_feat_idx[edge_id]
        if auto_create:
            # expand edge feature matrix
            assert self.edge_feature.shape[0] == len(self.edge_id_to_edge_feat_idx), (
                "edge feature map len != edge_id_to_edge_feat_idx"
                f"{self.edge_feature.shape[0]} vs"
                f"{len(self.edge_id_to_edge_feat_idx)}"
            )

            new_edge_feat_idx = self.edge_feature.shape[0]
            self.edge_id_to_edge_feat_idx[edge_id] = new_edge_feat_idx
            self._expand_edge_feature(new_edge_feat_idx)
            return self.edge_id_to_edge_feat_idx[edge_id]

        raise KeyError(f"no edge_id:{edge_id} in self.edge_id_to_edge_feat_idx")

    def _expand_edge_feature(self, edge_feat_idx: int):
        """expand edge feature matrix with np.nan"""
        _feat = np.zeros((1, self.edge_feat_dim), dtype=float)
        _feat.fill(np.nan)
        self.edge_feature = np.vstack((self.edge_feature, _feat))
        assert self.edge_feature.shape[0] - 1 == edge_feat_idx, (
            f"new edge_feat_idx {edge_feat_idx}!= edge_feature.shape[0]-1"
            f"({self.edge_feature.shape[0]}-1)"
        )

    def _update_edges_feature(
        self,
        update_timestamp: float,
        edge_features_override: List[Tuple[Any, Any, np.ndarray]],
        *args,
        **kwargs,
    ):
        # assume that src, tgt already exists.
        # hash edge and modify edge feature (src,tgt) -> id
        # edge_feat[id] = new_feat

        for e_iter, edge_feat_override in enumerate(edge_features_override):
            src, is_src_new = self.get_node_id(edge_feat_override[0])
            assert (
                not is_src_new
            ), f"src node must exist, current is_src_new: {is_src_new}"
            tgt, is_tgt_new = self.get_node_id(edge_feat_override[1])
            assert (
                not is_tgt_new
            ), f"src node must exist, current is_src_new: {is_src_new}"
            edge_id = self.get_edge_id((src, tgt))
            override_edge_feat = edge_feat_override[2]
            assert override_edge_feat.shape[0] == self.edge_feat_dim, (
                f"override_edge_feat dim ({override_edge_feat.shape[0]}) != "
                f"override_edge_feat dim ({self.edge_feat_dim})"
            )

            # override features
            edge_feat_idx = self.map_edge_id_to_edge_mat_idx(
                edge_id,
                auto_create=True,
            )
            self.edge_feature[edge_feat_idx, :] = override_edge_feat
        return None

    def _update_edges_label(
        self, update_timestamp: float, edge_events, *args, **kwargs
    ):
        """Not implemented yet."""
        raise NotImplementedError

    def _update_node_label(
        self,
        update_timestamp: float,
        node_label_override: List[Tuple[Any, int]],
        *args,
        **kwargs,
    ):
        """update node feature with node label override list.

        Args:
            update_timestamp (float): the graph batch update timestamp
            node_label_override (List[Tuple[Any, int]]): the node label
            override dictionary, 1st element is node raw desc.

        """
        for e_iter, node_lb_override in enumerate(node_label_override):
            node_raw_desc = node_lb_override[0]
            override_node_label = node_lb_override[1]
            node_id, is_id_new = self.get_node_id(
                node_raw_desc,
                auto_create=False,
            )
            assert not is_id_new, f"node ({node_raw_desc}) was unseen."
            node_label_idx = node_id  # reuse node_id as label vec idx.

            assert node_label_idx <= self.node_labels.shape[0] - 1, (
                f"node_label_idx ({node_label_idx}) should <="
                f" {self.node_labels.shape[0] - 1}"
            )

            assert isinstance(override_node_label, np.ndarray), (
                "override_node_label must be np.ndarray (Bool). "
                f"however, it is {type(override_node_label)}"
            )

            # update node label
            self.node_labels[node_label_idx] = override_node_label

    def _update_node_feature(
        self,
        update_timestamp: float,
        node_feature_override: List[Tuple[Any, np.ndarray]],
        *args,
        **kwargs,
    ):
        """update node feature with node feature override list.

        Args:
            update_timestamp (float): the graph batch update timestamp
            node_feature_override (List[Tuple[Any, np.ndarray]]): the
                node feature override dictionary, 1st element is node
                raw desc.
        """
        for e_iter, node_feat_override in enumerate(node_feature_override):
            node_raw_desc = node_feat_override[0]
            override_node_feat = node_feat_override[1]
            node_id, is_id_new = self.get_node_id(
                node_raw_desc,
                auto_create=False,
            )
            assert not is_id_new, f"node ({node_raw_desc}) was unseen."
            node_feat_idx = node_id  # reuse node_id as feat mat idx.

            assert node_feat_idx <= self.node_feature.shape[0] - 1, (
                f"node_feat_idx ({node_feat_idx}) should <="
                f" {self.node_feature.shape[0] - 1}"
            )

            assert override_node_feat.shape[0] == self.node_feat_dim, (
                f"override_node_feat dim ({override_node_feat.shape[0]}) !="
                f"node_feat_dim ({self.node_feat_dim})"
            )

            self.node_feature[node_feat_idx, :] = override_node_feat

    @staticmethod
    def __array_expand_helper(
        arr: np.ndarray,
        new_id: int,
        max_node_num: int,
        fill_in_val: float = np.nan,
    ) -> np.ndarray:
        """expand a vector along one axis with np.nan
        the new_id should be the index of the expanded element.

        Args:
            arr (np.ndarray): the expandable array
            new_id (int): the new idx to expand
            max_node_num (int): the max number of nodes
            fill_in_val (float): the init value for the expanded element

        Returns:
            np.ndarray: the expanded arr (vector)
        """
        if arr.shape[0] == max_node_num:
            return arr
        else:
            arr = np.hstack((arr, fill_in_val))
            assert (
                arr.shape[0] - 1 == new_id
            ), f"new node_id {new_id}!= arr.shape[0]-1 ({arr.shape[0]}-1)"
            return arr

    @staticmethod
    def apply_degree_add(base_val: float, add_val: float) -> float:
        """apply accumulated degree changes with degree>=0 check"""
        new_degree_val = base_val + add_val
        assert new_degree_val > 0, (
            "degree must be non-negative, current:"
            f"{new_degree_val}={base_val} + {add_val}"
        )
        return base_val + add_val
