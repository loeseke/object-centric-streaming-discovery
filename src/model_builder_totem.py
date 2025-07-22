"""\
Classes and functionality for discovering TOTeM model from its streaming representation and assessing its accuracy compared to offline model.
__author__: "Nina Löseke"
"""

from typing import Tuple, Any, Union
import graphviz
import itertools
import pandas as pd
import os
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
from vars import *
from model_buffers import TotemBuffer
from cache_policy_buffers import CachePolicy
from priority_policy_buffers import PPBCustom, PPBEventsPerObjectType, PPBLifespanPerObject, PPBLifespanPerObjectType, PPBObjectsPerEvent, PPBObjectsPerObjectType, PPBStridePerObject, PPBStridePerObjectType, PrioPolicyOrder
from utils import EventStream
from pathlib import Path
from offline_totem import discover_totem_offline


# Global variables useful for TOTeM-model building
ALL_ECS_VIS = [
    EC_ZERO,
    EC_ONE,
    EC_ZERO_ONE,
    EC_ONE_MANY,
    EC_ZERO_MANY
]
EC_TO_RANK = dict(zip(ALL_ECS_VIS + [None], range(len(ALL_ECS_VIS) + 1)))
MAX_EC_DIST = len(ALL_ECS_VIS)

ALL_LCS_VIS = [
    LC_ONE,
    LC_ZERO_ONE,
    LC_ONE_MANY,
    LC_ZERO_MANY
]
LC_TO_RANK = dict(zip(ALL_LCS_VIS + [None], range(len(ALL_LCS_VIS) + 1)))
MAX_LC_DIST = len(ALL_LCS_VIS)

ALL_TRS_VIS = [
    TR_DURING,
    TR_DURING_INVERSE,
    TR_PRECEDES,
    TR_PRECEDES_INVERSE,
    TR_PARALLEL
]
TR_TO_RANK = dict(zip(ALL_TRS_VIS + [None], range(len(ALL_TRS_VIS) + 1)))
MAX_TR_DIST = len(ALL_TRS_VIS)

TOTEM_MODEL_NAME = 'totem'
OCDFG_MODEL_NAME = 'ocdfg'

TR_TO_EDGE_ARROWHEAD = {
    TR_DURING: 'box',
    TR_DURING_INVERSE: 'obox',
    TR_PRECEDES: 'normal',
    TR_PRECEDES_INVERSE: 'onormal',
    TR_PARALLEL: 'teetee'
}
TR_EDGE_ATTR = {
    'arrowtail': 'none', 
    'dir': 'both', 
    'decorate': 'false'
}


class TotemModel(object):
    """
    Represents Temporal Object Type Model.

    Attributes
    ----------
    tau : float
        Filter parameter as minimum support for temporal, event-cardinality, and log-cardinality relations.
    model : TotemBuffer
        Streaming TOTeM from which TotemModel is discovered.
    log : str
        Path to log from which offline TotemModel is discovered.
    connected_types : set[Tuple[str, str]]
        Pairs of object types that co-occur in events or object-to-object relations.
    ots_to_ec_scores : dict[Tuple[str, str], dict[str, float]]
        Mapping of connected-type pairs to scores of event cardinalities.
    ots_to_lc_scores : dict[Tuple[str, str], dict[str, float]]
        Mapping of connected-type pairs to scores of log cardinalities.
    ots_to_tr_scores : dict[Tuple[str, str], dict[str, float]]
        Mapping of connected-type pairs to scores of temporal relations.
    ots_to_lc_oid_pairs : dict[Tuple[str, str], set[Tuple[str, str]]]
        Mapping of connected-type pairs from LC buffer to co-occurring objects.
    nodes : set[str]
        Nodes of TOTeM model corresponding to object types.
    edges : dict[str, dict[str, Any]]
        Directed edges of TOTeM model indicating connected object types.
    """
    
    def __init__(self, model_or_log : Union[TotemBuffer, str], tau : float = 1, verbose : bool = False) -> None:
        """
        Initializes a TotemModel object.

        Parameters
        ----------
        model_or_log : Union[TotemBuffer, str]
            Streaming TOTeM model or path to offline log from which TotemModel is discovered.
        tau : float, default=1
            Threshold for determining most accurate temporal, event-cardinality, and log-cardinality relations.
        verbose : bool, default=False
            If enabled, discovered TOTeM model is printed to terminal.

        Raises
        ------
        RuntimeError
            If neither streaming representation nor log to discover TOTeM model from are specified correctly.
        """
        self.tau : float = tau
        self.model : TotemBuffer = None
        self.log : str = None

        self.connected_types : set[Tuple[str, str]] = set()
        self.ots_to_ec_scores : dict[Tuple[str, str], dict[str, float]] = dict()
        self.ots_to_lc_scores : dict[Tuple[str, str], dict[str, float]] = dict()
        self.ots_to_lc_oid_pairs : dict[Tuple[str, str], set[Tuple[str, str]]] = dict()
        self.ots_to_tr_scores : dict[Tuple[str, str], dict[str, float]] = dict()

        self.nodes : set[str] = set()
        self.edges : dict[str, dict[str, Any]] = dict()

        if isinstance(model_or_log, TotemBuffer):
            self.model = model_or_log
        
            self.__evaluate_connected_types(model_or_log)
            self.__evaluate_ec(model_or_log)
            self.__evaluate_lc(model_or_log)
            self.__evaluate_tr(model_or_log)
            self.__build_totem_graph(tau, verbose)

        elif isinstance(model_or_log, str):
            self.log = model_or_log
            self.nodes, self.edges = discover_totem_offline(model_or_log, tau=tau, verbose=verbose)
        
        else:
            raise RuntimeError(f'Cannot build TotemModel from input of type {type(model_or_log)}.')

    def get_graphviz_file_name(self) -> str:
        """
        Derives unique file name for visualization output of TOTeM model that encodes its discovery parameters, e.g. coupled removal.
        
        Returns
        -------
        str
        """
        if self.model is not None:
            return f"cr-{int(self.model.coupled_removal)}_tau-{self.tau}.pdf"
        else:
            return f"offline_tau-{self.tau}.pdf"
    
    def get_graphviz_subdir(self) -> Path:
        """
        Derives unique path name for visualization output of TOTeM model that encodes its streaming parameters, e.g. temporal-relation buffer size, cache policy.
        
        Returns
        -------
        Path
            Subdirectory to save graphviz output to. If TOTeM is discovered offline, "totem" subdirectory is used.
        """
        if self.model is not None:
            pp_name = f'{self.model.pp_buf.prio_order.value.lower()}-{self.model.pp_buf.pp.value.lower().replace(' ', '-')}' if self.model.pp_buf is not None else 'none'

            return Path("totem",
                        f"{self.model.tr_buf_size}_{self.model.ec_buf_size}_{self.model.lc_buf_size}",
                        f"{self.model.cp.value.lower()}",
                        f"{pp_name}")
        else:
            return Path("totem")
    
    def get_graphviz_overlap_file_name(self, offl_tau : float) -> str:
        """
        Derives unique file name for visualization output of online OC-DFG overlapped with its corresponding offline OC-DFG.

        Parameters
        ----------
        offl_tau : float
            Filter parameter for relations in offline TOTeM model; should match filter parameter of online model.

        Returns
        -------
        str
        """
        return f"cr-{int(self.model.coupled_removal)}_tau-{self.tau}_vs_offline_tau-{offl_tau}.pdf"

    def __evaluate_connected_types(self, model : TotemBuffer) -> None:
        """Extracts connected types from streaming TOTeM."""
        # Return list of directed type tuples that are connected via shared event or O2O update based on non-OID buffers
        ot_pairs_ec_buf = [tuple(sorted((ot_a, ot_b))) for (ot_a, ot_b) in model.ec_buf.buf if ot_b]
        ot_pairs_lc_buf = [tuple(sorted((ot_a, ot_b))) for (ot_a, ot_b) in model.lc_buf.buf if ot_b]
        unique_ot_pairs = set(ot_pairs_ec_buf + ot_pairs_lc_buf)
        self.connected_types = unique_ot_pairs.copy()

        # Add reverse direction of OT pair
        for ot_a, ot_b in unique_ot_pairs:
            self.connected_types.add((ot_b, ot_a))
    
    def __evaluate_ec(self, model : TotemBuffer) -> None:
        """Scores event cardinalities for connected types based on EC buffer."""
        ots_to_events = dict()
        ot_to_unique_events = dict()
        ot_to_num_events = dict()

        for (ot_a, ot_b) in model.ec_buf.buf:
            ots_to_events.setdefault((ot_a, ot_b), dict())
            ot_to_unique_events.setdefault(ot_a, set())

            for ec_dict in model.ec_buf.buf[(ot_a, ot_b)]:
                ots_to_events[(ot_a, ot_b)][ec_dict[EVENT_ID]] = ec_dict[EVENT_CARD]
                ot_to_unique_events[ot_a].add(ec_dict[EVENT_ID])
        
        ot_to_num_events = {ot: len(unique_events) for ot, unique_events in ot_to_unique_events.items()}

        ots_to_ec_scores = dict()
        for ot_a, ot_b in self.connected_types:
            ots_to_ec_scores[(ot_a, ot_b)] = dict()
            if ot_a not in ot_to_num_events:
                for ec in ALL_ECS_VIS:
                    ots_to_ec_scores[(ot_a, ot_b)][ec] = None
            else:
                num_events_ot_a = ot_to_num_events[ot_a]
                ec_zero_ot_a = num_events_ot_a - (len(ots_to_events[(ot_a, ot_b)]) if (ot_a, ot_b) in ots_to_events else 0)
                ots_to_ec_scores[(ot_a, ot_b)][EC_ZERO] = ec_zero_ot_a
                # Set default value for ECs 1 and 1...* (e.g. if OT pair has no associated events and therefore only scores 1 for EC 0 and 0...1, 0...*)
                ots_to_ec_scores[(ot_a, ot_b)][EC_ONE] = 0
                ots_to_ec_scores[(ot_a, ot_b)][EC_ZERO_ONE] = ec_zero_ot_a
                ots_to_ec_scores[(ot_a, ot_b)][EC_ONE_MANY] = 0
                ots_to_ec_scores[(ot_a, ot_b)][EC_ZERO_MANY] = ec_zero_ot_a

                if (ot_a, ot_b) in ots_to_events:
                    for _, ec_cards in ots_to_events[(ot_a, ot_b)].items():
                        for ec in ec_cards:
                            if ec == EC_ZERO:
                                continue 
                            else:
                                ots_to_ec_scores[(ot_a, ot_b)][ec] += 1
                
                # Divide counts of event cardinalities by number of events with non-empty set of objects of type a to obtain score
                ots_to_ec_scores[(ot_a, ot_b)] = {ec: count/num_events_ot_a for ec, count in ots_to_ec_scores[(ot_a, ot_b)].items()}
        
        self.ots_to_ec_scores = ots_to_ec_scores

    def __evaluate_lc(self, model : TotemBuffer) -> None:
        """Scores log cardinalities for connected types based on LC buffer."""
        ots_to_lc_oid_pairs = dict()
        ot_to_lc_oids = dict()
        for (ot_a, ot_b), oid_pair_dicts in model.lc_buf.buf.items():
            for oid_pair_dict in oid_pair_dicts:
                oid_a, oid_b = oid_pair_dict[OBJECT_PAIR]

                ots_to_lc_oid_pairs.setdefault((ot_a, ot_b), set())
                ots_to_lc_oid_pairs[(ot_a, ot_b)].add((oid_a, oid_b)) 

                ots_to_lc_oid_pairs.setdefault((ot_b, ot_a), set())
                ots_to_lc_oid_pairs[(ot_b, ot_a)].add((oid_b, oid_a))

                ot_to_lc_oids.setdefault(ot_a, set())
                ot_to_lc_oids[ot_a].add(oid_a)

                ot_to_lc_oids.setdefault(ot_b, set())
                ot_to_lc_oids[ot_b].add(oid_b)
        
        ots_to_lc_scores = dict()
        for ot_a, ot_b in self.connected_types:
            ots_to_lc_scores[(ot_a, ot_b)] = dict()
            if (ot_a, ot_b) not in ots_to_lc_oid_pairs:
                for lc in ALL_LCS_VIS:
                    ots_to_lc_scores[(ot_a, ot_b)][lc] = None
            else:
                for lc in ALL_LCS_VIS:
                    ots_to_lc_scores[(ot_a, ot_b)][lc] = 0

                num_oids_ot_a = len(ot_to_lc_oids[ot_a])

                for ot_a_oid in ot_to_lc_oids[ot_a]:
                    matching_oid_pairs = [(x, y) for (x, y) in ots_to_lc_oid_pairs[(ot_a, ot_b)] if x == ot_a_oid]
                    num_matches_ot_a_oid = len(matching_oid_pairs)

                    if num_matches_ot_a_oid == 0:
                        ots_to_lc_scores[(ot_a, ot_b)][LC_ZERO_ONE] += 1
                        ots_to_lc_scores[(ot_a, ot_b)][LC_ZERO_MANY] += 1
                    elif num_matches_ot_a_oid == 1:
                        ots_to_lc_scores[(ot_a, ot_b)][LC_ONE] += 1
                        ots_to_lc_scores[(ot_a, ot_b)][LC_ZERO_ONE] += 1
                        ots_to_lc_scores[(ot_a, ot_b)][LC_ONE_MANY] += 1
                        ots_to_lc_scores[(ot_a, ot_b)][LC_ZERO_MANY] += 1
                    else:
                        ots_to_lc_scores[(ot_a, ot_b)][LC_ONE_MANY] += 1
                        ots_to_lc_scores[(ot_a, ot_b)][LC_ZERO_MANY] += 1
                
                # Divide by number of unique objects of source OT to obtain LC score
                ots_to_lc_scores[(ot_a, ot_b)] = {lc: count/num_oids_ot_a for lc, count in ots_to_lc_scores[(ot_a, ot_b)].items()}

        self.ots_to_lc_scores = ots_to_lc_scores
        self.ots_to_lc_oid_pairs = ots_to_lc_oid_pairs

    def __evaluate_tr(self, model : TotemBuffer) -> None:
        """Scores temporal relations for connected types based on TR and LC buffer."""
        # Compute TR scores for connected OTs and respective connected (via LC buffer) OIDs in TR buffer
        ots_to_tr_scores = dict()
        for ot_a, ot_b in self.connected_types:
            if (ot_a, ot_b) in self.ots_to_lc_oid_pairs:
                num_oid_pairs = len(self.ots_to_lc_oid_pairs[(ot_a, ot_b)])
                
                ots_to_tr_scores[(ot_a, ot_b)] = {tr: 0 for tr in ALL_TRS_VIS}
                
                for oid_a, oid_b in self.ots_to_lc_oid_pairs[(ot_a, ot_b)]:
                    if oid_a not in model.tr_buf.buf or oid_b not in model.tr_buf.buf:
                        continue

                    a_min = model.tr_buf.buf[oid_a][FIRST_SEEN]
                    a_max = model.tr_buf.buf[oid_a][LAST_SEEN]
                    b_min = model.tr_buf.buf[oid_b][FIRST_SEEN]
                    b_max = model.tr_buf.buf[oid_b][LAST_SEEN]

                    if b_min <= a_min <= a_max <= b_max:
                        ots_to_tr_scores[(ot_a, ot_b)][TR_DURING] += 1
                    if a_min <= b_min <= b_max <= a_max:
                        ots_to_tr_scores[(ot_a, ot_b)][TR_DURING_INVERSE] += 1
                    if a_min <= a_max <= b_min <= b_max or a_min < b_min <= a_max < b_max:
                        ots_to_tr_scores[(ot_a, ot_b)][TR_PRECEDES] += 1
                    if b_min <= b_max <= a_min <= a_max or b_min < a_min <= b_max < a_max:
                        ots_to_tr_scores[(ot_a, ot_b)][TR_PRECEDES_INVERSE] += 1

                # Normalize each TR score to range from 0 to 1
                ots_to_tr_scores[(ot_a, ot_b)] = {tr: count/num_oid_pairs for tr, count in ots_to_tr_scores[(ot_a, ot_b)].items()}
                
                # TR_PARALLEL always holds
                ots_to_tr_scores[(ot_a, ot_b)][TR_PARALLEL] = 1.0
            else:
                # NOTE: ot_a and ot_b might be connected just via EC buffer, but have no matching OID pairs in LC buffer to derive exact TR from; in this case, set TR_PARALLEL score to 1.0
                ots_to_tr_scores[(ot_a, ot_b)] = {tr: 0 for tr in ALL_TRS_VIS}
                ots_to_tr_scores[(ot_a, ot_b)][TR_PARALLEL] = 1.0

        self.ots_to_tr_scores = ots_to_tr_scores
    
    def __build_totem_graph(self, tau : float = 1, verbose : bool = False) -> None:
        """Discovers TOTeM graph with most precise relations according to given filter parameter tau."""
        for ot_a, ot_b in self.connected_types:
            self.nodes.add(ot_a)
            self.nodes.add(ot_b)
            
            tr_annot = None
            lc_annot = None
            ec_annot = None
            
            for tr in ALL_TRS_VIS:
                if self.ots_to_tr_scores[(ot_a, ot_b)][tr] and self.ots_to_tr_scores[(ot_a, ot_b)][tr] >= tau:
                    tr_annot = tr
                    break

            for lc in ALL_LCS_VIS:
                if self.ots_to_lc_scores[(ot_a, ot_b)][lc] and self.ots_to_lc_scores[(ot_a, ot_b)][lc] >= tau:
                    lc_annot = lc
                    break
            
            for ec in ALL_ECS_VIS:
                if self.ots_to_ec_scores[(ot_a, ot_b)][ec] and self.ots_to_ec_scores[(ot_a, ot_b)][ec] >= tau:
                    ec_annot = ec
                    break
            
            # Add edge for connected OT pair (edges added both ways)
            self.edges[(ot_a, ot_b)] = {'TR': tr_annot, 'LC': lc_annot, 'EC': ec_annot}
        
            if verbose:
                print(f'{ot_a} -> {ot_b}:\t TR: {tr_annot}\t LC: {lc_annot}\t EC: {ec_annot}')

    def visualize(self, output_dir : Path, output_file : str, ot_to_hex_color : dict[str, Any]) -> None:
        """
        Draws discovered, annotated TOTeM model via graphviz. Object types are highlighted in the given colors.
        Given output directory is created automatically if it does not yet exist.

        Parameters
        ----------
        output_dir : Path
            Output directory to which PDF is saved.
        output_file : str
            Name of file to which PDF is saved.
        ot_to_hex_color : dict[str, Any]
            Mapping of object types in the OC-DFG to hex color codes.

        Returns
        -------
        None
        """
        G = graphviz.Digraph(graph_attr={'label': f'Filter parameter: tau = {self.tau}\nEdge annotation: LC (EC)', 
                                         'fontname': GV_FONT, 'fontsize': GV_GRAPH_FONTSIZE,
                                         'margin': '0.1,0.1', 
                                         'overlap': 'false',
                                         'rankdir': 'LR'})
        
        for ot in self.nodes:
            G.node(ot, label=ot, shape='box', fontname=GV_FONT, fontsize=GV_NODE_FONTSIZE, color=ot_to_hex_color[ot])
        
        for (ot_a, ot_b), edge_dict in self.edges.items():
            G.edge(ot_a, ot_b, 
                   label=f'{edge_dict['LC']}\n({edge_dict['EC']})',
                   fontname=GV_FONT, fontsize=GV_EDGE_FONTSIZE,
                   arrowhead=TR_TO_EDGE_ARROWHEAD[edge_dict['TR']],
                   **TR_EDGE_ATTR)

        os.makedirs(output_dir, exist_ok=True)
        G.render(filename='tmp', cleanup=True, format='pdf')
        os.replace('tmp.pdf', output_dir / output_file)


def get_totem_accuracy(offline : TotemModel, online : TotemModel) -> dict[str, float]:
    """
    Computes evaluation metrics to assess quality of online model compared to offline model, e.g. accuracy, precision, and recall for connected types and relations between them.

    Parameters
    ----------
    offline : TotemModel
        TOTeM model discovered offline from full log.
    online : TotemModel
        TOTeM model discovered from streaming representation.
    
    Returns
    -------
    dict[str, float]
        Mapping of evaluation metrics to values.
    """
    # Compute node accuracy based on confusion matrix (concept of "negatives" does not apply to nodes)
    onl_pos_nodes = online.nodes
    offl_pos_nodes = offline.nodes
    all_nodes = onl_pos_nodes.union(offl_pos_nodes)
    onl_neg_nodes = all_nodes - onl_pos_nodes
    offl_neg_nodes = all_nodes - offl_pos_nodes
    node_tp = len(onl_pos_nodes.intersection(offl_pos_nodes))
    node_fp = len(onl_pos_nodes - offl_pos_nodes)
    node_tn = len(onl_neg_nodes.intersection(offl_neg_nodes))
    node_fn = len(onl_neg_nodes - offl_neg_nodes)

    node_acc = (node_tp + node_tn) / (node_tp + node_fp + node_tn + node_fn) if node_tp + node_fp + node_tn + node_fn > 0 else 0
    node_prec = node_tp / (node_tp + node_fp) if node_tp + node_fp > 0 else 0
    node_rec = node_tp / (node_tp + node_fn) if node_tp + node_fn > 0 else 0

    # Compute arc accuracy based on confusion matrix
    possible_arcs = set(itertools.product(all_nodes, all_nodes))
    onl_pos_arcs = set(online.edges.keys())
    onl_neg_arcs = possible_arcs - onl_pos_arcs
    offl_pos_arcs = set(offline.edges.keys())
    offl_neg_arcs = possible_arcs - offl_pos_arcs

    arc_tp = len(onl_pos_arcs.intersection(offl_pos_arcs))
    arc_fp = len(onl_pos_arcs - offl_pos_arcs)
    arc_tn = len(onl_neg_arcs.intersection(offl_neg_arcs))
    arc_fn = len(onl_neg_arcs - offl_neg_arcs)

    arc_acc = (arc_tp + arc_tn) / (arc_tp + arc_tn + arc_fp + arc_fn) if arc_tp + arc_tn + arc_fp + arc_fn > 0 else 0
    arc_prec = arc_tp / (arc_tp + arc_fp) if arc_tp + arc_fp > 0 else 0
    arc_rec = arc_tp / (arc_tp + arc_fn) if arc_tp + arc_fn > 0 else 0

    # Compute distance-based accuracy of arc annotations
    shared_arcs = onl_pos_arcs.intersection(offl_pos_arcs)
    num_shared_arcs = len(shared_arcs)
    tr_err = sum([abs(TR_TO_RANK[online.edges[arc]['TR']] - TR_TO_RANK[offline.edges[arc]['TR']]) for arc in shared_arcs])
    max_possible_tr_err = num_shared_arcs * MAX_TR_DIST
    tr_annot_acc = 1 - tr_err / max_possible_tr_err if max_possible_tr_err > 0 else 0

    ec_err = sum([abs(EC_TO_RANK[online.edges[arc]['EC']] - EC_TO_RANK[offline.edges[arc]['EC']]) for arc in shared_arcs])
    max_possible_ec_err = num_shared_arcs * MAX_EC_DIST
    ec_annot_acc = 1 - ec_err / max_possible_ec_err if max_possible_ec_err > 0 else 0
    
    lc_err = sum([abs(LC_TO_RANK[online.edges[arc]['LC']] - LC_TO_RANK[offline.edges[arc]['LC']]) for arc in shared_arcs])
    max_possible_lc_err = num_shared_arcs * MAX_LC_DIST
    lc_annot_acc = 1 - lc_err / max_possible_lc_err if max_possible_lc_err > 0 else 0

    # Compute "binary" accuracy of arc annotations
    # tr_err = 0.0
    # ec_err = 0.0
    # lc_err = 0.0
    # for arc in shared_arcs:
    #     if online.edges[arc]['TR'] == offline.edges[arc]['TR']:
    #         tr_err += 1
    #     if online.edges[arc]['LC'] == offline.edges[arc]['LC']:
    #         lc_err += 1
    #     if online.edges[arc]['EC'] == offline.edges[arc]['EC']:
    #         ec_err += 1
    # tr_annot_acc = tr_err / num_shared_arcs
    # ec_annot_acc = ec_err / num_shared_arcs
    # lc_annot_acc = lc_err / num_shared_arcs

    return {'node recall': node_rec,
            'node accuracy': node_acc,
            'node precision': node_prec,
            'arc recall': arc_rec,
            'arc accuracy': arc_acc,
            'arc precision': arc_prec,
            'TR accuracy': tr_annot_acc,
            'EC accuracy': ec_annot_acc,
            'LC accuracy': lc_annot_acc}


def get_totem_avg_scores(offline : TotemModel, online : TotemModel) -> dict[str, float]:
    """
    Averages TOTeM model accuracy, precision, and recall across source/sink nodes/arcs and inner nodes/arcs.

    Parameters
    ----------
    offline : TotemModel
        TOTeM model discovered offline from full log.
    online : TotemModel
        TOTeM model discovered from streaming representation.

    Returns
    -------
    dict[str, float]
        Mapping of evaluation metrics to values.
    """
    scores = get_totem_accuracy(offline, online)
    rec_scores = [scores['node recall'], scores['arc recall']]
    prec_scores = [scores['node precision'], scores['arc precision']]
    acc_scores = [scores['node accuracy'], scores['arc accuracy']]
    filtered_rec_scores = [rec for rec in rec_scores if rec is not None]
    filtered_prec_scores = [prec for prec in prec_scores if prec is not None]
    filtered_acc_scores = [acc for acc in acc_scores if acc is not None]
    
    return {'recall': np.mean(filtered_rec_scores) if len(filtered_rec_scores) > 0 else 0,
            'precision': np.mean(filtered_prec_scores) if len(filtered_prec_scores) > 0 else 0,
            'accuracy': np.mean(filtered_acc_scores) if len(filtered_acc_scores) > 0 else 0}


def visualize_totem_overlap(offline : TotemModel, online : TotemModel, output_dir : Path, output_file : str, ot_to_hex_color : dict[str, Any]) -> None:
    """
    Draws online TOTeM model overlapped with offline corresponding TOTeM model via graphviz. Object types are highlighted in the given colors.
    Differences in temporal relations and event/log cardinalities are annotated. Given output directory is created automatically if it does not yet exist.

    Parameters
    ----------
    offline : TotemModel
        TOTeM model discovered offline from full log.
    online : TotemModel
        TOTeM model discovered from streaming representation.
    output_dir : Path
        Output directory to which PDF is saved.
    output_file : str
        Name of file to which PDF is saved.
    ot_to_hex_color : dict[str, Any]
        Mapping of object types in the TOTeM model to hex color codes.

    Returns
    -------
    None
    """
    shared_nodes = offline.nodes.intersection(online.nodes)
    offl_nodes = offline.nodes - online.nodes
    onl_nodes = online.nodes - offline.nodes
    all_onl_arcs = set(online.edges.keys())
    all_offl_arcs = set(offline.edges.keys())
    onl_arcs = all_onl_arcs - all_offl_arcs
    offl_arcs = all_offl_arcs - all_onl_arcs
    shared_arcs = all_onl_arcs.intersection(all_offl_arcs)
    
    G = graphviz.Digraph(graph_attr={'label': f'Line style: solid -> shared, dashed -> offline, dotted -> online\nEdge-annotation & edge-arrowhead position: top -> offline, bottom -> online\nEdge annotation: LC (EC)',
                                     'fontname': GV_FONT, 'fontsize': GV_GRAPH_FONTSIZE,
                                     'margin': '0.1,0.1', 
                                     'overlap': 'false',
                                     'rankdir': 'LR'})
        
    for ot in shared_nodes:
        G.node(ot, label=ot, shape='box', fontname=GV_FONT, fontsize=GV_NODE_FONTSIZE, style='solid', color=ot_to_hex_color[ot])
    for ot in offl_nodes:
        G.node(ot, label=ot, shape='box', fontname=GV_FONT, fontsize=GV_NODE_FONTSIZE, style='dashed', color=ot_to_hex_color[ot])
    for ot in onl_nodes:
        G.node(ot, label=ot, shape='box', fontname=GV_FONT, fontsize=GV_NODE_FONTSIZE, style='dotted', color=ot_to_hex_color[ot])
    
    for (ot_a, ot_b) in shared_arcs:
        onl_tr = online.edges[(ot_a, ot_b)]['TR']
        offl_tr = offline.edges[(ot_a, ot_b)]['TR']
        shared_arrowhead = TR_TO_EDGE_ARROWHEAD[offl_tr]
        if onl_tr != offl_tr:
            shared_arrowhead += TR_TO_EDGE_ARROWHEAD[onl_tr]
        
        G.edge(ot_a, ot_b, 
               style='solid',
               label=f'{offline.edges[(ot_a, ot_b)]['LC']} ({offline.edges[(ot_a, ot_b)]['EC']})\n{online.edges[(ot_a, ot_b)]['LC']} ({online.edges[(ot_a, ot_b)]['EC']})',
               fontname=GV_FONT, fontsize=GV_EDGE_FONTSIZE,
               arrowhead=shared_arrowhead,
               **TR_EDGE_ATTR)
        
    for (ot_a, ot_b) in offl_arcs:
        G.edge(ot_a, ot_b, 
               style='dashed',
               label=f'{offline.edges[(ot_a, ot_b)]['LC']} ({offline.edges[(ot_a, ot_b)]['EC']})',
               fontname=GV_FONT, fontsize=GV_EDGE_FONTSIZE,
               arrowhead=TR_TO_EDGE_ARROWHEAD[offline.edges[(ot_a, ot_b)]['TR']],
               **TR_EDGE_ATTR)
    
    for (ot_a, ot_b) in onl_arcs:
        G.edge(ot_a, ot_b, 
               style='dotted',
               label=f'{online.edges[(ot_a, ot_b)]['LC']} ({online.edges[(ot_a, ot_b)]['EC']})',
               fontname=GV_FONT, fontsize=GV_EDGE_FONTSIZE,
               arrowhead=TR_TO_EDGE_ARROWHEAD[online.edges[(ot_a, ot_b)]['TR']],
               **TR_EDGE_ATTR)

    os.makedirs(output_dir, exist_ok=True)
    G.render(filename='tmp', cleanup=True, format='pdf')
    os.replace('tmp.pdf', output_dir / output_file)