"""\
Classes and functionality for discovering (OC-)DFG from its streaming representation and assessing its accuracy compared to offline model.
__author__: "Nina Löseke"
"""

from typing import Tuple, Any, Union
import graphviz
import itertools
import pandas as pd
import networkx as nx
import os
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
from vars import *
from model_buffers import OcdfgBuffer, OcdfgBufferPerObjectType
from cache_policy_buffers import CachePolicy
from priority_policy_buffers import PPBCustom, PPBEventsPerObjectType, PPBLifespanPerObject, PPBLifespanPerObjectType, PPBObjectsPerEvent, PPBObjectsPerObjectType, PPBStridePerObject, PPBStridePerObjectType, PrioPolicyOrder
from utils import EventStream, td_to_str
from pathlib import Path
from offline_ocdfg import discover_ocdfg_offline


class DfgModel(object):
    """
    Represents Directly-Follows Graph.

    Attributes
    ----------
    ot : str
        Object type for which DFG is discovered.
    normalize : bool
        If set to True, node and arc frequencies are normalized to [0, 1] range across entire DFG. If False, absolute frequencies are used.
    nodes : dict[str, Any]
        Mapping of activities to frequencies.
    arcs : dict[Tuple[str, str], Any]
        Mapping of directly-follows relations (activity-activity pairs) to frequencies.
    arcs_to_avg_dur : dict[Tuple[str, str], Any]
        Mapping of directly-follows relations (activity-activity pairs) to average durations.
    source_sink_nodes : set[str]
        Source and sink node for given object type.
    source_sink_arcs : set[Tuple[str, str]]
        Arcs connecting source node to initial activity and connecting final activities to sink node.
    """

    def __init__(self, arc_buf_or_dfg_dict : Union[dict[Tuple[str, str], dict[str, Any]], dict[str, Any]], ot : str, normalize : bool = False, dict_as_dfg : bool = False, verbose : bool = False) -> None:
        """
        Initializes DfgModel object.

        Parameters
        ----------
        arc_buf_or_dfg_dict : Union[dict[Tuple[str, str], dict[str, Any], dict[str, Any]]
            DFG arc buffer from which DFG is discovered or DFG defined as dictionary (used to compare online to offline model).
        ot : str
            Object type for which DFG is discovered.
        normalize : bool, default=False
            If set to True, node and arc frequencies are normalized to [0, 1] range across entire DFG. If False, absolute frequencies are used.
        dict_as_dfg : bool, default=False
            Indicates if DFG is specified as dictionary (True) or DFG arc buffer (False).
        verbose : bool, default=False
            If enabled, discovered DFG is printed to terminal.
        """
        self.ot = ot
        self.normalize = normalize
    
        self.nodes : dict[str, Any] = dict()
        self.arcs : dict[Tuple[str, str], Any] = dict()
        self.arcs_to_avg_dur : dict[Tuple[str, str], Any] = dict()
        self.source_sink_nodes : set[str] = set()
        self.source_sink_arcs : set[Tuple[str, str]] = set()

        if not dict_as_dfg:
            self.__build_inner_nodes_arcs(arc_buf_or_dfg_dict)
            self.__init_source_sink(arc_buf_or_dfg_dict)
            self.build_source_sink()
        else:
            self.nodes = arc_buf_or_dfg_dict['nodes']
            self.arcs = arc_buf_or_dfg_dict['arcs']
            self.arcs_to_avg_dur = arc_buf_or_dfg_dict['arcs_to_avg_dur']
            self.source_sink_nodes = arc_buf_or_dfg_dict['source_sink_nodes']
            self.source_sink_arcs = arc_buf_or_dfg_dict['source_sink_arcs']
        
        if self.normalize:
            self.nodes = self.__normalize_node_frequencies(self.nodes)
            self.arcs = self.__normalize_arc_frequencies(self.arcs)
        
        if verbose:
            print(self)
    
    def __normalize_node_frequencies(self) -> None:
        """Normalizes node frequencies to [0, 1] range."""
        if len(self.nodes) > 0:
            new_sum_node_freq = sum(self.nodes.values())
            self.nodes = {node: freq/new_sum_node_freq for node, freq in self.nodes.items()}
    
    def __normalize_arc_frequencies(self) -> None:
        """Normalizes arc frequencies to [0, 1] range."""
        if len(self.arcs) > 0:
            new_sum_arc_freq = sum(self.arcs.values())
            self.arcs = {arc: freq/new_sum_arc_freq for arc, freq in self.arcs.items()}

    def __init_source_sink(self, arc_buf : dict[Tuple[str, str], dict[str, Any]]) -> None:
        """Defines source and sink nodes and initializes source arcs for initial activities in DFG arc buffer."""
        # Insert OT source and sink nodes
        source_name = f'Source {self.ot}'
        self.source_sink_nodes.add(source_name)
        self.source_sink_nodes.add(f'Sink {self.ot}')

        # Get initial arcs that are in arc buffer
        init_arcs_in_buf = arc_buf.keys() - self.arcs.keys()

        # Insert initial arcs according to (None, <activity>) already in arc buffer
        for (x, y) in init_arcs_in_buf:
            # Ensure initial activities from arc buffer are part of node set
            if y in self.nodes:
                self.source_sink_arcs.add((source_name, y))

    def build_source_sink(self, clean_up : bool = False) -> None:
        """
        Inserts source and sink nodes and arcs.
        Source arcs are inserted for activities without predecessors. Sink arcs are inserted from activities that have no successor.
        Additional source/sink arcs are inserted for most frequent activities that are part of isolated cycles.

        Parameters
        ----------
        clean_up : bool, default=False
            Removes superfluous source/sink arcs in case activities were removed.
        
        Returns
        -------
        None
        """
        source_name = f'Source {self.ot}'
        sink_name = f'Sink {self.ot}'

        # Insert arcs to initial activities that don't already have incoming arcs
        nodes_w_in_arcs = set()
        nodes_w_out_arcs = set()
        for (x, y) in self.arcs.keys():
            nodes_w_in_arcs.add(y)
            nodes_w_out_arcs.add(x)

        init_activities = set(self.nodes.keys()) - nodes_w_in_arcs
        for init_act in init_activities:
            self.source_sink_arcs.add((source_name, init_act))
        
        # Insert arcs from final activities to sink node
        final_activities = set(self.nodes.keys()) - nodes_w_out_arcs
        for final_act in final_activities:
            self.source_sink_arcs.add((final_act, sink_name))
        
        # Find cycles in inner arcs and connect them to source and sink if not already done
        arcs_wo_source_sink = [(x, y) for (x, y) in self.arcs.keys() if x not in init_activities and y not in final_activities]
        G_nx = nx.DiGraph(arcs_wo_source_sink)
        gen_cycles = sorted(nx.simple_cycles(G_nx))

        # Check for arc entering/leaving cycle
        for cycle in gen_cycles:
            cycle_in_arcs = [(x, y) for (x, y) in self.arcs.keys() if x not in cycle and y in cycle]
            cycle_source_arcs = [(x, y) for (x, y) in self.source_sink_arcs if x not in cycle and y in cycle]
            cycle_out_arcs = [(x, y) for (x, y) in self.arcs.keys() if x in cycle and y not in cycle]
            cycle_sink_arcs = [(x, y) for (x, y) in self.source_sink_arcs if x in cycle and y not in cycle]

            # NOTE: insert sink/source arc for cycle at most frequent node of cycle if cycle doesn't having in-coming/out-going inner or source/sink arcs
            if len(cycle_in_arcs) == 0 or len(cycle_out_arcs) == 0:
                most_freq_cycle_node = max([(node, self.nodes[node]) for node in cycle], key=lambda tup: tup[1])[0]
                if len(cycle_in_arcs) == 0 and len(cycle_source_arcs) == 0:
                    self.source_sink_arcs.add((source_name, most_freq_cycle_node))
                if len(cycle_out_arcs) == 0 and len(cycle_sink_arcs) == 0:
                    self.source_sink_arcs.add((most_freq_cycle_node, sink_name))
        
        if clean_up:
            # Remove source/sink arcs that are no longer needed, e.g. arc between inner node that no longer exists and sink
            for x, y in list(self.source_sink_arcs):
                if (x == source_name and y not in self.nodes) or (x not in self.nodes and y == sink_name):
                    self.source_sink_arcs.discard((x, y))

    def __build_inner_nodes_arcs(self, arc_buf : dict[Tuple[str, str], dict[str, Any]]) -> None:
        """Derives non-source/sink nodes and arcs from DFG arc buffer. Computes average arc durations."""
        # Use target activities from arc buffer as nodes, e.g. start activities from (None, <start activity>) arcs
        nodes = dict()
        for (act_a, act_b), arc_dict_list in arc_buf.items():
            for arc_dict in arc_dict_list:
                nodes[act_b] = nodes.get(act_b, 0) + arc_dict[TARGET_ACTIVITY_FREQ]

        # Extract "inner" arcs from arc buffer
        # NOTE: need to check "act_a in nodes" in case there is arc w/ "left-over" initial activity that's otherwise not buffered and therefore has no frequency/duration info
        inner_arcs = [(act_a, act_b) for act_a, act_b in arc_buf if act_a is not None and act_a in nodes]
        inner_arc_to_count = {tup: len(arc_buf[tup]) for tup in inner_arcs}
        inner_arc_to_total_min = {}
        for (act_a, act_b), arc_dict_list in arc_buf.items():
            for arc_dict in arc_dict_list:
                if (act_a, act_b) in inner_arcs:
                    # Convert arc duration to minutes before adding up to avoid potential overflow errors
                    inner_arc_to_total_min[(act_a, act_b)] = inner_arc_to_total_min.get((act_a, act_b), 0) + arc_dict[ACTIVITY_DURATION].total_seconds() / 60

        arcs = {tup: inner_arc_to_count[tup] for tup in inner_arcs}
        # Round average arc duration mathematically to minutes
        arcs_to_avg_dur = {tup: np.mean([arc_dict[ACTIVITY_DURATION] for arc_dict in arc_buf[tup]]) for tup in inner_arcs}

        self.nodes = nodes
        self.arcs = arcs
        self.arcs_to_avg_dur = arcs_to_avg_dur
    
    def __str__(self) -> str:
        """Creates string listing all DFG components."""
        return_str = ''
        return_str += f'DFG for object type {self.ot}:\n'
        return_str += f'Nodes: {self.nodes}\n'
        return_str += f'Arcs: {self.arcs}\n'
        return_str += f'Arc durations: {[(arc, str(td.to_pytimedelta())) for arc, td in self.arcs_to_avg_dur.items()]}\n'
        return_str += f'Sink/source nodes: {self.source_sink_nodes}\n'
        return_str += f'Sink/source arcs: {self.source_sink_arcs}\n'
        return return_str
    
    def visualize(self, output_dir : Path, output_file : str) -> None:
        """
        Draws discovered, annotated DFG via graphviz.
        Given output directory is created automatically if it does not yet exist.

        Parameters
        ----------
        output_dir : Path
            Output directory to which PDF is saved.
        output_file : str
            Name of file to which PDF is saved.

        Returns
        -------
        None
        """
        G = graphviz.Digraph(graph_attr={'label': f'Normalized node/edge frequencies: {self.normalize}\nEdge annotation: top -> frequency, bottom -> avg. duration [days-hours:min]', 
                                         'fontname': GV_FONT, 'fontsize': GV_GRAPH_FONTSIZE,
                                         'margin': '0.1,0.1', 
                                         'overlap': 'false',
                                         'rankdir': 'LR'})

        for act, act_freq in self.nodes.items():
            G.node(act, 
                   label=f'{act}\n{act_freq:.2f}', 
                   shape='box', 
                   fontname=GV_FONT, 
                   fontsize=GV_NODE_FONTSIZE, 
                   color='black')
        
        for (act_a, act_b), inner_arc_freq in self.arcs.items():
            G.edge(act_a, act_b, 
                   label=f'{inner_arc_freq:.2f}\n{td_to_str(self.arcs_to_avg_dur[(act_a, act_b)])}',
                   fontname=GV_FONT, 
                   fontsize=GV_EDGE_FONTSIZE, 
                   color='black')

        for source_sink in self.source_sink_nodes:
            G.node(source_sink, 
                   label=source_sink, 
                   shape='oval', 
                   fontname=GV_FONT, 
                   fontsize=GV_NODE_FONTSIZE, 
                   color='black')

        for (x, y) in self.source_sink_arcs:
            G.edge(x, y)

        os.makedirs(output_dir, exist_ok=True)
        G.render(filename='tmp', cleanup=True, format='pdf')
        os.replace('tmp.pdf', output_dir / output_file)


def get_new_gap_arcs(
        nodes_to_prune : list[Any], 
        arcs : dict[Tuple[Any, Any], float], 
        arcs_to_avg_dur : dict[Tuple[Any, Any], pd.Timedelta]
    ) -> list[Tuple[Any, Any, float, pd.Timedelta]]:
    """
    Identifies new inner arcs to insert or update when nodes are pruned from a DFG to cover "gaps". 
    For example, if node b is removed from a sequence (a, b, c), then a new arc from a to c is inserted or, if it already exists, its frequency and duration are updated.

    Parameters
    ----------
    nodes_to_prune : list[Any]
        Nodes to remove from DFG.
    arcs : dict[Tuple[Any, Any], float]
        Mapping of activity-activity pairs to frequencies.
    arcs_to_avg_dur : dict[Tuple[Any, Any], pd.Timedelta]
        Mapping of activity-activity pairs to average durations.

    Returns
    -------
    list[Tuple[Any, Any, float, pd.Timedelta]]
        List of new or updated arcs and associated frequency and average duration.
    """
    # Find potential new arcs spanning gap of (chain of) nodes that are pruned from graph
    new_gap_arcs = list()

    # Reduce arcs to nodes to prune
    arcs_betw_nodes_to_prune = [(x, y) for x, y in arcs if x in nodes_to_prune and y in nodes_to_prune]
    pruned_paths = list()
    G_nodes_to_prune = nx.DiGraph(arcs_betw_nodes_to_prune)
    for pruned_source, pruned_target in itertools.product(nodes_to_prune, nodes_to_prune):
        if G_nodes_to_prune.has_node(pruned_source) and G_nodes_to_prune.has_node(pruned_target):
            paths_betw_nodes_to_prune = nx.all_simple_paths(G_nodes_to_prune, pruned_source, pruned_target)
            pruned_paths += paths_betw_nodes_to_prune
    
    for path in pruned_paths:
        pred_path = [x for (x, y) in arcs.keys() if y == path[0] and x not in nodes_to_prune]
        succ_path = [y for (x, y) in arcs.keys() if x == path[-1] and y not in nodes_to_prune]

        for (a, b) in list(itertools.product(pred_path, succ_path)):
            # Set frequency of new arc to minimum of frequencies along removed node(s) and non-pruned arcs entering/leaving gap of removed node(s)
            new_freq = min([arcs[(a, path[0])], arcs[(path[-1], b)]] + [arcs[(path[i], path[i+1])] for i in range(len(path)-1)])

            new_path_tds = [arcs_to_avg_dur[(a, path[0])], arcs_to_avg_dur[(path[-1], b)]] + [arcs_to_avg_dur[(path[i], path[i+1])] for i in range(len(path)-1)]
            new_td = pd.Timedelta(0)
            for td in new_path_tds:
                new_td += td
            new_gap_arcs.append((a, b, new_freq, new_td))

    return new_gap_arcs


class OcdfgModel(object):
    """
    Represents Object-Centric Directly-Follows Graph.

    Attributes
    ----------
    normalize : bool
        If set to True, node and arc frequencies are normalized to [0, 1] range across entire OC-DFG. If False, absolute frequencies are used.
    model : Union[OcdfgBuffer, OcdfgBufferPerObjectType]
        Streaming OC-DFG from which OcdfgModel is discovered.
    log : str
        Path to log from which offline OcdfgModel is discovered.
    prune_node_frac : float
        Fraction of least frequent nodes to prune.
    prune_arc_freq : float
        Fraction of least frequent arcs to prune.
    pruned_nodes : bool
        Flag to indicate if nodes have already been pruned.
    pruned_arcs : bool
        Flag to indicate if arcs have already been pruned.
    dfgs_per_ot : dict[str, DfgModel]
        Mapping of object types to discovered DFGs.
    nodes : dict[str, dict[str, float]]
        Mapping of nodes to dictionaries of relevant object types and their frequencies.
    arcs : dict[Tuple[str, str], dict[str, float]]
        Mapping of arcs to dictionaries of relevant object types and their frequencies.
    arcs_to_avg_dur : dict[Tuple[str, str], dict[str, pd.Timedelta]]
        Mapping of arcs to dictionaries of relevant object types and their average durations.
    source_sink_nodes : dict[str, set[str]]
        Mapping of object types to their source/sink nodes.
    source_sink_arcs : dict[str, set[Tuple[str, str]]]
        Mapping of object types to their source/sink arcs.
    """

    def __init__(self, model_or_log : Union[OcdfgBuffer, OcdfgBufferPerObjectType, str], prune_node_frac : float = 0.0, prune_arc_frac : float = 0.0, normalize : bool = False, verbose : bool = False) -> None:
        """
        Initializes an OcdfgModel object.

        Parameters
        ----------
        model_or_log : Union[OcdfgBuffer, OcdfgBufferPerObjectType, str]
            Streaming OC-DFG or path to offline log from which OcdfgModel is discovered.
        prune_node_frac : float, default=0.0
            Fraction of least frequent nodes to prune.
        prune_arc_freq : float, default=0.0
            Fraction of least frequent arcs to prune.
        normalize : bool, default=False
            If set to True, node and arc frequencies are normalized to [0, 1] range across entire OC-DFG. If False, absolute frequencies are used.
        verbose : bool, default=False
            If enabled, discovered OC-DFG is printed to terminal.
        """
        self.normalize = normalize
        self.model = None
        self.log = None
        
        if not (0 <= prune_node_frac <= 1):
            raise ValueError(f'Removal fraction for pruning OC-DFG nodes must be between 0 and 1.')
        if not (0 <= prune_arc_frac <= 1):
            raise ValueError(f'Removal fraction for pruning OC-DFG arcs must be between 0 and 1.')
        self.prune_node_frac = prune_node_frac
        self.prune_arc_frac = prune_arc_frac
        self.pruned_nodes = False
        self.pruned_arcs = False

        self.dfgs_per_ot : dict[str, DfgModel] = dict()
        self.nodes : dict[str, dict[str, float]] = dict()
        self.arcs : dict[Tuple[str, str], dict[str, float]]  = dict()
        self.arcs_to_avg_dur : dict[Tuple[str, str], dict[str, pd.Timedelta]] = dict()
        self.source_sink_nodes : dict[str, set[str]] = dict()
        self.source_sink_arcs : dict[str, set[Tuple[str, str]]] = dict()

        if isinstance(model_or_log, (OcdfgBuffer, OcdfgBufferPerObjectType)):
            self.model = model_or_log
            self.__build_dfgs_per_ot(model_or_log, verbose=False)
            self.__merge_dfgs()

            # Normalize "inner" node/arc frequencies according to their fraction among all nodes/arcs
            if self.normalize:
                self.__normalize_node_frequencies()
                self.__normalize_arc_frequencies()
                
        elif isinstance(model_or_log, str):
            self.log = model_or_log
            ocdfg_dict, dfg_per_ot_dicts = discover_ocdfg_offline(model_or_log)

            for ot in dfg_per_ot_dicts:
                self.dfgs_per_ot[ot] = DfgModel(dfg_per_ot_dicts[ot], ot, dict_as_dfg=True)

            self.nodes = ocdfg_dict['nodes']
            self.arcs = ocdfg_dict['arcs']
            self.arcs_to_avg_dur = ocdfg_dict['arcs_to_avg_dur']
            self.source_sink_nodes = ocdfg_dict['source_sink_nodes']
            self.source_sink_arcs = ocdfg_dict['source_sink_arcs']

            # Normalize "inner" node/arc frequencies according to their fraction among all nodes/arcs
            if self.normalize:
                self.__normalize_node_frequencies()
                self.__normalize_arc_frequencies()
            
        else:
            raise RuntimeError(f'Cannot build OcdfgModel from input of type {type(model_or_log)}.')

        if self.prune_node_frac > 0:
            self.__prune_nodes_by_freq()
        if self.prune_arc_frac > 0:
            self.__prune_arcs_by_freq()

        if verbose:
            print(self)
    
    def get_graphviz_subdir(self) -> Path:
        """
        Derives unique path name for visualization output of OC-DFG that encodes its streaming parameters, e.g. node-buffer size, cache policy.
        
        Returns
        -------
        Path
            Subdirectory to save graphviz output to. If OC-DFG is discovered offline, "ocdfg" subdirectory is used.
        """
        if self.model is not None:
            pp_name = f'{self.model.pp_buf.prio_order.value.lower()}-{self.model.pp_buf.pp.value.lower().replace(' ', '-')}' if self.model.pp_buf is not None else 'none'
            model_buf_name = "ocdfg" if isinstance(self.model, OcdfgBuffer) else "ocdfg-per-ot"

            return Path(model_buf_name,
                        f"{self.model.node_buf_size}_{self.model.arc_buf_size}",
                        f"{self.model.cp.value.lower()}",
                        f"{pp_name}")
        else:
            return Path('ocdfg')
    
    def get_graphviz_file_name(self) -> str:
        """
        Derives unique file name for visualization output of OC-DFG that encodes its discovery parameters, e.g. coupled removal, fraction of nodes to prune.
        
        Returns
        -------
        str
        """
        if self.model is not None:
            file_name = f"cr-{int(self.model.coupled_removal)}_rmn-{self.prune_node_frac}_rma-{self.prune_arc_frac}"
        else:
            file_name = f"offline_rmn-{self.prune_node_frac}_rma-{self.prune_arc_frac}"
        
        if self.normalize:
            file_name += '_normalized'
        return file_name + ".pdf"


    def get_graphviz_overlap_file_name(self, offl_prune_node_frac : float, offl_prune_arc_frac : float) -> str:
        """
        Derives unique file name for visualization output of online OC-DFG overlapped with its corresponding offline OC-DFG.

        Parameters
        ----------
        offl_prune_node_frac : float
            Fraction of least frequent nodes in offline model to prune; should match online model.
        offl_prune_arc_frac : float
            Fraction of least frequent arcs in offline model to prune; should match online model.

        Returns
        -------
        str
        """
        return f"cr-{int(self.model.coupled_removal)}_rmn-{self.prune_node_frac}_rma-{self.prune_arc_frac}_vs_offline_rmn-{offl_prune_node_frac}_rma-{offl_prune_arc_frac}.pdf"
    
    def __normalize_node_frequencies(self) -> None:
        """Normalizes node frequencies to [0, 1] range."""
        sum_node_freq = 0.0
        for node in self.nodes:
            for ot in self.nodes[node]:
                sum_node_freq += self.nodes[node][ot]
        
        # Normalize inner node frequencies by new total sum of node values
        for node in self.nodes:
            for ot in self.nodes[node]:
                self.nodes[node][ot] /= sum_node_freq
    
    def __normalize_arc_frequencies(self) -> None:
        """Normalizes arc frequencies to [0, 1] range."""
        sum_arc_freq = 0.0
        for arc in self.arcs:
           for ot in self.arcs[arc]:
                sum_arc_freq += self.arcs[arc][ot]

        # Normalize inner arc frequencies by new total sum of arc values
        for arc in self.arcs:
            for ot in self.arcs[arc]:
                self.arcs[arc][ot] /= sum_arc_freq
    
    def __build_dfgs_per_ot(self, model : OcdfgBuffer | OcdfgBufferPerObjectType, verbose : bool = False) -> None:
        """Discovers DfgModels for each object type from given streaming OC-DFG."""
        if isinstance(model, OcdfgBuffer):
            # Filter arc buffer based on OT to build each DFG from
            ots = self.model.arc_buf.get_buffered_nested_values_for_key(OBJECT_TYPE)
            ots_to_filtered_arc_buf = {ot: dict() for ot in ots}
            for arc, arc_dict_list in list(self.model.arc_buf.buf.items()):
                for i, arc_dict in enumerate(arc_dict_list):
                    ot = arc_dict[OBJECT_TYPE]
                    ots_to_filtered_arc_buf[ot].setdefault(arc, list())
                    arc_dict.pop(OBJECT_TYPE)
                    ots_to_filtered_arc_buf[ot][arc].append(arc_dict)
        
        elif isinstance(model, OcdfgBufferPerObjectType):
            ots_to_filtered_arc_buf = {ot: self.model.dfg_bufs[ot].arc_buf.buf.copy() for ot in self.model.dfg_bufs}
        
        else:
            raise NotImplementedError(f'Cannot build OcdfgModel from model buffer of type {type(model)}.')
        
        dfgs_per_ot = {ot: DfgModel(arc_buf, ot, normalize=False, verbose=verbose) for ot, arc_buf in ots_to_filtered_arc_buf.items()}
        self.dfgs_per_ot = dfgs_per_ot
    
    def __reset_inner_nodes_arcs(self) -> None:
        """Clears all inner nodes and arcs."""
        self.nodes = dict()
        self.arcs = dict()
        self.arcs_to_avg_dur = dict()
    
    def __merge_dfgs(self) -> None:
        """Merges DfgModels into OcdfgModel where nodes belong to several object types and arcs are typed."""
        for ot, dfg in self.dfgs_per_ot.items():
            # Create set of nodes from all DFGs per object type
            for dfg_node, freq in dfg.nodes.items():
                if dfg_node in self.nodes:
                    self.nodes[dfg_node][ot] = freq
                else:
                    self.nodes[dfg_node] = {ot: freq}
            
            # Create set of arcs from all DFGs per object type
            for dfg_arc, freq in dfg.arcs.items():
                if dfg_arc in self.arcs:
                    self.arcs[dfg_arc][ot] = freq
                else:
                    self.arcs[dfg_arc] = {ot: freq}
            
            for dfg_arc, td in dfg.arcs_to_avg_dur.items():
                if dfg_arc in self.arcs_to_avg_dur:
                    self.arcs_to_avg_dur[dfg_arc][ot] = td
                else:
                    self.arcs_to_avg_dur[dfg_arc] = {ot: td}
            
            # Add sink/source nodes for object type and their corresponding arcs
            self.source_sink_nodes[ot] = dfg.source_sink_nodes.copy()
            self.source_sink_arcs[ot] = dfg.source_sink_arcs.copy()
    
    def __prune_arcs_by_freq(self) -> None:
        """Removes least frequent arcs from OC-DFG."""
        if self.pruned_arcs:
            print(f'OC-DFG arcs already pruned w/ removal fraction {self.prune_arc_frac}. No repeated pruning is done.')
            return

        # NOTE: consider arcs per object type to determine least frequent arcs opposed to based on total arc weight across all associated object types
        n_arcs = len(self.arcs.values())
        n_rm = round(n_arcs*self.prune_arc_frac)
        if n_rm == 0:
            return

        # Unroll arc dictionary to list of tuples for sorting individual arcs
        arc_tuples = list()
        for arc in self.arcs:
            for ot, freq in self.arcs[arc].items():
                arc_tuples.append((freq, ot, arc))
        
        arc_tuples_asc = sorted(arc_tuples, key=lambda tup: tup[0], reverse=False)
        arc_tuples_rm = arc_tuples_asc[:n_rm]
        
        # Remove inner arcs from corresponding DFGs and re-merge DFGs to re-set source/sink arcs
        for _, ot, arc in arc_tuples_rm:
            self.dfgs_per_ot[ot].arcs.pop(arc)
            self.dfgs_per_ot[ot].arcs_to_avg_dur.pop(arc)
        
        for ot, dfg in self.dfgs_per_ot.items():
            dfg.build_source_sink(clean_up=True)
        
        self.__reset_inner_nodes_arcs()
        self.__merge_dfgs()

        # Normalize "inner" arc frequencies according to their fraction among all arcs ("inner" node frequencies unchanged)
        if self.normalize:
            self.__normalize_arc_frequencies()
            self.__normalize_node_frequencies()

        self.pruned_arcs = True

    def __prune_nodes_by_freq(self) -> None:
        """Removes least frequent nodes from OC-DFG and fills in gaps with new arcs where possible."""
        if self.pruned_nodes:
            print(f'OC-DFG nodes already pruned w/ removal fraction {self.prune_node_frac}. No repeated pruning is done.')
            return

        n_nodes = len(self.nodes)
        n_rm = round(n_nodes*self.prune_node_frac)
        if n_rm == 0:
            return
        
        # Unroll node dictionary to list of tuples for sorting individual nodes
        oc_nodes_to_total_freq = {node: sum(ot_dict.values()) for node, ot_dict in self.nodes.items()}
        oc_nodes_rm = sorted(oc_nodes_to_total_freq.items(), key=lambda tup: tup[1], reverse=False)[:n_rm]
        oc_nodes_rm = [tup[0] for tup in oc_nodes_rm]
        nodes_rm = dict()
        for node in oc_nodes_rm:
            for ot in self.nodes[node]:
                nodes_rm.setdefault(ot, list())
                nodes_rm[ot].append(node)

        arcs_rm = dict()
        for (x, y), ot_dict in self.arcs.items():
            if x in oc_nodes_rm or y in oc_nodes_rm:
                for ot in ot_dict:
                    if ot in arcs_rm:
                        arcs_rm[ot].append((x, y))
                    else:
                        arcs_rm[ot] = [(x, y)]
        
        # Add new arcs "bridging" gaps caused by removed nodes per object type
        for ot in nodes_rm:
            dfg = self.dfgs_per_ot[ot]
            new_gap_arcs = get_new_gap_arcs(nodes_rm[ot], dfg.arcs, dfg.arcs_to_avg_dur)
            gap_arc_to_count = {(x, y): int((x, y) in dfg.arcs) for x, y, _, _ in new_gap_arcs}

            for new_x, new_y, new_freq, new_td in new_gap_arcs:
                gap_arc_to_count[(new_x, new_y)] += 1
                if (new_x, new_y) in dfg.arcs:
                    dfg.arcs[(new_x, new_y)] += new_freq
                    dfg.arcs_to_avg_dur[(new_x, new_y)] += new_td
                else:
                    dfg.arcs[(new_x, new_y)] = new_freq
                    dfg.arcs_to_avg_dur[(new_x, new_y)] = new_td
            
            # Compute new average duration based on # new and old occurences of arc for OT
            for new_x, new_y in set([(x, y) for x, y, _, _ in new_gap_arcs]):
                dfg.arcs_to_avg_dur[(new_x, new_y)] /= gap_arc_to_count[(new_x, new_y)]
        
        # Remove nodes and arcs and refresh DFGs of affected OTs
        all_affected_ots = nodes_rm.keys()
        for ot, affected_nodes in nodes_rm.items():
            for affected_node in affected_nodes:
                self.dfgs_per_ot[ot].nodes.pop(affected_node, None)
            if ot in arcs_rm:
                for affected_arc in arcs_rm[ot]:
                    self.dfgs_per_ot[ot].arcs.pop(affected_arc, None)
                    self.dfgs_per_ot[ot].arcs_to_avg_dur.pop(affected_arc, None)
        for ot in all_affected_ots:
            self.dfgs_per_ot[ot].build_source_sink(clean_up=True)

        # Rebuild merged OC-DFG from refreshed DFGs
        self.__reset_inner_nodes_arcs()
        self.__merge_dfgs()

        if self.normalize:
            self.__normalize_arc_frequencies()
            self.__normalize_node_frequencies()
        
        self.pruned_nodes = True

    def __str__(self) -> str:
        """Creates string listing all OC-DFG components."""
        res_str = f'\nMerged OC-DFG nodes:\n'
        for i, (node, freq_dict) in enumerate(self.nodes.items()):
            rounded_freq_dict = {key: round(val, 2) for key, val in freq_dict.items()}
            rounded_freq_dict = dict(sorted(rounded_freq_dict.items()))
            res_str += f'{i+1}. {node}\t{rounded_freq_dict}\n'

        res_str += f'Merged OC-DFG arcs:\n'
        for i, (arc, freq_dict) in enumerate(self.arcs.items()):
            rounded_freq_dict = {key: round(val, 2) for key, val in freq_dict.items()}
            rounded_freq_dict = dict(sorted(rounded_freq_dict.items()))
            res_str += f'{i+1}. {arc}\t{rounded_freq_dict}\n'
        
        res_str += f'Merged OC-DFG arc durations:\n'
        for i, (arc, dur_dict) in enumerate(self.arcs_to_avg_dur.items()):
            dur_dict = dict(sorted(dur_dict.items()))
            dur_dict = {ot: str(dur) for ot, dur in dur_dict.items()}
            res_str += f'{i+1}. {arc}\t{dur_dict}\n'
        
        res_str += f'Merged OC-DFG source/sink nodes:\n'
        for i, (ot, source_sink_nodes) in enumerate(self.source_sink_nodes.items()):
            res_str += f'{i+1}. {ot}: {source_sink_nodes}\n'
        
        res_str += f'Merged OC-DFG source/sink arcs:\n'
        for i, (ot, source_sink_arcs) in enumerate(self.source_sink_arcs.items()):
            res_str += f'{i+1}. {ot}: {source_sink_arcs}\n'

        return res_str

    def visualize(self, output_dir : Path, output_file : str, ot_to_hex_color : dict[str, Any], visualize_dfgs : bool = True) -> None:
        """
        Draws discovered, annotated OC-DFG via graphviz. Object types are highlighted in the given colors.
        Given output directory is created automatically if it does not yet exist.

        Parameters
        ----------
        output_dir : Path
            Output directory to which PDF is saved.
        output_file : str
            Name of file to which PDF is saved.
        ot_to_hex_color : dict[str, Any]
            Mapping of object types in the OC-DFG to hex color codes.
        visualize_dfgs : bool, default=True
            If True, individual DFGs for given OC-DFG are also drawn.

        Returns
        -------
        None
        """
        G = graphviz.Digraph(graph_attr={'label': f'Normalized node/edge frequencies: {self.normalize}\nPruning: {self.prune_node_frac*100:.0f}% of least frequent nodes, then {self.prune_arc_frac*100:.0f}% of least frequent arcs pruned\nEdge annotation: top -> frequency, bottom -> avg. duration [days-hours:min]', 
                                         'fontname': GV_FONT, 'fontsize': GV_GRAPH_FONTSIZE,
                                         'margin': '0.1,0.1', 
                                         'overlap': 'false',
                                         'rankdir': 'LR'})

        for act, act_freq_dict in self.nodes.items():
            total_act_freq = sum(act_freq_dict.values())
            G.node(act, 
                   label=f'{act}\n{total_act_freq:.2f}', 
                   shape='box', 
                   fontname=GV_FONT, 
                   fontsize=GV_NODE_FONTSIZE, 
                   color='black')
        
        for (act_a, act_b), arc_freq_dict in self.arcs.items():
            for ot, arc_freq in arc_freq_dict.items():
                G.edge(act_a, act_b, 
                       label=f'{arc_freq:.2f}\n{td_to_str(self.arcs_to_avg_dur[(act_a, act_b)][ot])}',
                       fontname=GV_FONT,
                       fontsize=GV_EDGE_FONTSIZE,
                       color=ot_to_hex_color[ot])

        for ot, node_list in self.source_sink_nodes.items():
            for source_sink in node_list:
                G.node(source_sink, 
                    label=source_sink, 
                    shape='oval', 
                    fontname=GV_FONT, 
                    fontsize=GV_NODE_FONTSIZE, 
                    color=ot_to_hex_color[ot])

        for ot, arc_list in self.source_sink_arcs.items():
            for (x, y) in arc_list:
                G.edge(x, y, color=ot_to_hex_color[ot])
        
        os.makedirs(output_dir, exist_ok=True)
        G.render(filename='tmp', cleanup=True, format='pdf')
        os.replace('tmp.pdf', output_dir / output_file)

        if visualize_dfgs:
            for ot, dfg in self.dfgs_per_ot.items():
                if self.model is not None:
                    dfg_output_file = f'cr-{int(self.model.coupled_removal)}_rmn-{self.prune_node_frac}_rma-{self.prune_arc_frac}_ot-{ot.replace(' ', '-').lower()}.pdf'
                else:
                    dfg_output_file = f'offline_rmn-{self.prune_node_frac}_rma-{self.prune_arc_frac}_ot-{ot.replace(' ', '-').lower()}.pdf'
                dfg.visualize(output_dir, dfg_output_file)


def get_ocdfg_accuracy(offline: OcdfgModel, online : OcdfgModel) -> dict[str, float]:
    """
    Computes evaluation metrics to assess quality of online model compared to offline model, e.g. accuracy, precision, and recall for nodes and arcs.

    Parameters
    ----------
    offline : OcdfgModel
        OC-DFG discovered offline from full log.
    online : OcdfgModel
        OC-DFG discovered from streaming representation.
    
    Returns
    -------
    dict[str, float]
        Mapping of evaluation metrics to values.
    """
    # Compute activity-node accuracy & precision based on confusion matrix (concept of "negatives" does not apply to nodes)
    onl_pos_nodes = set(online.nodes.keys())
    offl_pos_nodes = set(offline.nodes.keys())
    possible_nodes = onl_pos_nodes.union(offl_pos_nodes)
    onl_neg_nodes = possible_nodes - onl_pos_nodes
    offl_neg_nodes = possible_nodes - offl_pos_nodes

    node_tp = len(onl_pos_nodes.intersection(offl_pos_nodes))
    node_fp = len(onl_pos_nodes - offl_pos_nodes)
    node_tn = len(onl_neg_nodes.intersection(offl_neg_nodes))
    node_fn = len(onl_neg_nodes - offl_neg_nodes)

    node_acc = (node_tp + node_tn) / (node_tp + node_tn + node_fp + node_fn) if node_tp + node_tn + node_fp + node_fn > 0 else 0
    node_prec = node_tp / (node_tp + node_fp) if node_tp + node_fp > 0 else 0
    node_rec = node_tp / (node_tp + node_fn) if node_tp + node_fn > 0 else 0

    # Compute inner arc accuracy & precision per OT based on confusion matrix
    onl_ots = set(online.source_sink_nodes.keys())
    offl_ots = set(offline.source_sink_nodes.keys())
    possible_ot_arcs = set()
    for ot in onl_ots:
        onl_ot_dfg_nodes = online.dfgs_per_ot[ot].nodes.keys()
        for ot_dfg_arc in itertools.product(onl_ot_dfg_nodes, onl_ot_dfg_nodes):
            possible_ot_arcs.add((ot, ot_dfg_arc))
    for ot in offl_ots:
        offl_ot_dfg_nodes = offline.dfgs_per_ot[ot].nodes.keys()
        for ot_dfg_arc in itertools.product(offl_ot_dfg_nodes, offl_ot_dfg_nodes):
            possible_ot_arcs.add((ot, ot_dfg_arc))

    onl_pos_ot_arcs = set()
    for arc, ot_dict in online.arcs.items():
        onl_pos_ot_arcs.update([(ot, arc) for ot in ot_dict])
    offl_pos_ot_arcs = set()
    for arc, ot_dict in offline.arcs.items():
        offl_pos_ot_arcs.update([(ot, arc) for ot in ot_dict])
    onl_neg_ot_arcs = possible_ot_arcs - onl_pos_ot_arcs
    offl_neg_ot_arcs = possible_ot_arcs - offl_pos_ot_arcs

    arc_fp = len(onl_pos_ot_arcs - offl_pos_ot_arcs)
    arc_tp = len(onl_pos_ot_arcs.intersection(offl_pos_ot_arcs))
    arc_fn = len(onl_neg_ot_arcs - offl_neg_ot_arcs)
    arc_tn = len(onl_neg_ot_arcs.intersection(offl_neg_ot_arcs))

    arc_acc = (arc_tp + arc_tn) / (arc_tp + arc_tn + arc_fp + arc_fn) if arc_tp + arc_tn + arc_fp + arc_fn > 0 else 0
    arc_prec = arc_tp / (arc_tp + arc_fp) if arc_tp + arc_fp > 0 else 0
    arc_rec = arc_tp / (arc_tp + arc_fn) if arc_tp + arc_fn > 0 else 0

    # Compute mean absolute error for total activity-node frequencies
    onl_nodes = set(online.nodes.keys())
    offl_nodes = set(offline.nodes.keys())
    shared_nodes = onl_nodes.intersection(offl_nodes)
    num_shared_nodes = len(shared_nodes)
    total_node_freq_err = 0.0
    for node in shared_nodes:
        total_node_freq_err += abs(sum(offline.nodes[node].values()) - sum(online.nodes[node].values()))
    
    total_node_freq_mae = total_node_freq_err / num_shared_nodes if num_shared_nodes > 0 else 0

    # Compute mean absolute error for inner arc frequencies
    onl_pos_arcs = set(online.arcs.keys())
    offl_pos_arcs = set(offline.arcs.keys())
    shared_arcs = onl_pos_arcs.intersection(offl_pos_arcs)
    num_shared_ot_arcs = 0
    arc_freq_err = 0.0
    for arc in shared_arcs:
        shared_arc_ots = set(offline.arcs[arc].keys()).intersection(set(online.arcs[arc].keys()))
        num_shared_ot_arcs += len(shared_arc_ots)
        for ot in shared_arc_ots:
            arc_freq_err += abs(offline.arcs[arc][ot] - online.arcs[arc][ot])
    
    arc_freq_mae = arc_freq_err / num_shared_ot_arcs if num_shared_ot_arcs > 0 else 0

    # Compute accuracy & precision of source/sink nodes based on confusion matrix
    onl_pos_source_sinks = set(itertools.chain.from_iterable(online.source_sink_nodes.values()))
    offl_pos_source_sinks = set(itertools.chain.from_iterable(offline.source_sink_nodes.values()))
    possible_source_sinks = onl_pos_source_sinks.union(offl_pos_source_sinks)
    onl_neg_source_sinks = possible_source_sinks - onl_pos_source_sinks
    offl_neg_source_sinks = possible_source_sinks - offl_pos_source_sinks

    source_sink_tp = len(onl_pos_source_sinks.intersection(offl_pos_source_sinks))
    source_sink_fp = len(onl_pos_source_sinks - offl_pos_source_sinks)
    source_sink_tn = len(onl_neg_source_sinks.intersection(offl_neg_source_sinks))
    source_sink_fn = len(onl_neg_source_sinks - offl_neg_source_sinks)

    source_sink_acc = (source_sink_tp + source_sink_tn) / (source_sink_tp + source_sink_tn + source_sink_fp + source_sink_fn) if source_sink_tp + source_sink_tn + source_sink_fp + source_sink_fn > 0 else 0
    source_sink_prec = source_sink_tp / (source_sink_tp + source_sink_fp) if source_sink_tp + source_sink_fp > 0 else 0
    source_sink_rec = source_sink_tp / (source_sink_tp + source_sink_fn) if source_sink_tp + source_sink_fn > 0 else 0

    # Compute accuracy & precision of source/sink arcs per OT based on confusion matrix
    onl_ots = set(online.source_sink_arcs.keys())
    offl_ots = set(offline.source_sink_arcs.keys())
    possible_source_sink_arcs = set()
    for ot in onl_ots:
        possible_source_sink_arcs.update([(ot, (f'Source {ot}', act_node)) for act_node in online.dfgs_per_ot[ot].nodes.keys()])
        possible_source_sink_arcs.update([(ot, (act_node, f'Sink {ot}')) for act_node in online.dfgs_per_ot[ot].nodes.keys()])
    for ot in offl_ots:
        possible_source_sink_arcs.update([(ot, (f'Source {ot}', act_node)) for act_node in offline.dfgs_per_ot[ot].nodes.keys()])
        possible_source_sink_arcs.update([(ot, (act_node, f'Sink {ot}')) for act_node in offline.dfgs_per_ot[ot].nodes.keys()])

    onl_source_sink_arcs = set()
    offl_source_sink_arcs = set()
    for ot, ot_source_sink_arcs in online.source_sink_arcs.items():
        onl_source_sink_arcs.update([(ot, ot_source_sink_arc) for ot_source_sink_arc in ot_source_sink_arcs])
    offl_source_sink_arcs = set()
    for ot, ot_source_sink_arcs in offline.source_sink_arcs.items():
        offl_source_sink_arcs.update([(ot, ot_source_sink_arc) for ot_source_sink_arc in ot_source_sink_arcs])
        
    source_sink_arc_tp = len(onl_source_sink_arcs.intersection(offl_source_sink_arcs))
    source_sink_arc_fp = len(onl_source_sink_arcs - offl_source_sink_arcs)
    offl_neg_source_sink_arcs = possible_source_sink_arcs - offl_source_sink_arcs
    onl_neg_source_sink_arcs = possible_source_sink_arcs - onl_source_sink_arcs
    source_sink_arc_tn = len(onl_neg_source_sink_arcs.intersection(offl_neg_source_sink_arcs))
    source_sink_arc_fn = len(onl_neg_source_sink_arcs - offl_neg_source_sink_arcs)

    source_sink_arc_acc = (source_sink_arc_tp + source_sink_arc_tn) / (source_sink_arc_tp + source_sink_arc_tn + source_sink_arc_fp + source_sink_arc_fn) if source_sink_arc_tp + source_sink_arc_tn + source_sink_arc_fp + source_sink_arc_fn > 0 else 0
    source_sink_arc_prec = source_sink_arc_tp / (source_sink_arc_tp + source_sink_arc_fp) if source_sink_arc_tp + source_sink_arc_fp > 0 else 0
    source_sink_arc_rec = source_sink_arc_tp / (source_sink_arc_tp + source_sink_arc_fn) if source_sink_arc_tp + source_sink_arc_fn > 0 else 0

    return {'node recall': node_rec,
            'node accuracy': node_acc,
            'node precision': node_prec,
            'arc recall': arc_rec,
            'arc accuracy': arc_acc,
            'arc precision': arc_prec,
            'total node freq. MAE': total_node_freq_mae,
            'arc freq. MAE': arc_freq_mae,
            'source/sink recall': source_sink_rec,
            'source/sink accuracy': source_sink_acc,
            'source/sink precision': source_sink_prec,
            'source/sink arc recall': source_sink_arc_rec,
            'source/sink arc accuracy': source_sink_arc_acc,
            'source/sink arc precision': source_sink_arc_prec}


def get_ocdfg_avg_scores(offline : OcdfgModel, online : OcdfgModel) -> dict[str, float]:
    """
    Averages OC-DFG accuracy, precision, and recall across source/sink nodes/arcs and inner nodes/arcs.

    Parameters
    ----------
    offline : OcdfgModel
        OC-DFG discovered offline from full log.
    online : OcdfgModel
        OC-DFG discovered from streaming representation.

    Returns
    -------
    dict[str, float]
        Mapping of evaluation metrics to values.
    """
    scores = get_ocdfg_accuracy(offline, online)
    rec_scores = [scores['node recall'], scores['arc recall'], scores['source/sink recall'], scores['source/sink arc recall']]
    prec_scores = [scores['node precision'], scores['arc precision'], scores['source/sink precision'], scores['source/sink arc precision']]
    acc_scores = [scores['node accuracy'], scores['arc accuracy'], scores['source/sink accuracy'], scores['source/sink arc accuracy']]
    filtered_rec_scores = [rec for rec in rec_scores if rec is not None]
    filtered_prec_scores = [prec for prec in prec_scores if prec is not None]
    filtered_acc_scores = [acc for acc in acc_scores if acc is not None]
    
    return {'recall': np.mean(filtered_rec_scores) if len(filtered_rec_scores) > 0 else 0,
            'precision': np.mean(filtered_prec_scores) if len(filtered_prec_scores) > 0 else 0,
            'accuracy': np.mean(filtered_acc_scores) if len(filtered_acc_scores) > 0 else 0}


def visualize_ocdfg_overlap(offline : OcdfgModel, online : OcdfgModel, output_dir : Path, output_file : str, ot_to_hex_color : dict[str, Any]) -> None:
    """
    Draws online OC-DFG overlapped with offline corresponding OC-DFG via graphviz. Object types are highlighted in the given colors.
    Differences in node and arc frequencies/durations are annotated. Given output directory is created automatically if it does not yet exist.

    Parameters
    ----------
    offline : OcdfgModel
        OC-DFG discovered offline from full log.
    online : OcdfgModel
        OC-DFG discovered from streaming representation.
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
    all_onl_nodes = set(online.nodes.keys())
    all_offl_nodes = set(offline.nodes.keys())
    shared_nodes = all_onl_nodes.intersection(all_offl_nodes)
    onl_nodes = all_onl_nodes - all_offl_nodes
    offl_nodes = all_offl_nodes - all_onl_nodes

    all_onl_ot_source_sinks = set()
    for ot, ot_source_sinks in online.source_sink_nodes.items():
        all_onl_ot_source_sinks.update([(ot, ot_source_sink) for ot_source_sink in ot_source_sinks])

    all_offl_ot_source_sinks = set()
    for ot, ot_source_sinks in offline.source_sink_nodes.items():
        all_offl_ot_source_sinks.update([(ot, ot_source_sink) for ot_source_sink in ot_source_sinks])

    shared_ot_source_sinks = all_onl_ot_source_sinks.intersection(all_offl_ot_source_sinks)
    onl_ot_source_sinks = all_onl_ot_source_sinks - all_offl_ot_source_sinks
    offl_ot_source_sinks = all_offl_ot_source_sinks - all_onl_ot_source_sinks

    all_onl_ot_arcs = set()
    for arc, ot_dict in online.arcs.items():
        all_onl_ot_arcs.update([(ot, arc) for ot in ot_dict])

    all_offl_ot_arcs = set()
    for arc, ot_dict in offline.arcs.items():
        all_offl_ot_arcs.update([(ot, arc) for ot in ot_dict])

    shared_ot_arcs = all_onl_ot_arcs.intersection(all_offl_ot_arcs)
    onl_ot_arcs = all_onl_ot_arcs - all_offl_ot_arcs
    offl_ot_arcs = all_offl_ot_arcs - all_onl_ot_arcs
    
    all_onl_ot_source_sink_arcs = set()
    for ot, ot_source_sink_arcs in online.source_sink_arcs.items():
        all_onl_ot_source_sink_arcs.update([(ot, ot_source_sink_arc) for ot_source_sink_arc in ot_source_sink_arcs])

    all_offl_ot_source_sink_arcs = set()
    for ot, ot_source_sink_arcs in offline.source_sink_arcs.items():
        all_offl_ot_source_sink_arcs.update([(ot, ot_source_sink_arc) for ot_source_sink_arc in ot_source_sink_arcs])

    shared_ot_source_sink_arcs = all_onl_ot_source_sink_arcs.intersection(all_offl_ot_source_sink_arcs)
    onl_ot_source_sink_arcs = all_onl_ot_source_sink_arcs - all_offl_ot_source_sink_arcs
    offl_ot_source_sink_arcs = all_offl_ot_source_sink_arcs - all_onl_ot_source_sink_arcs

    G = graphviz.Digraph(graph_attr={'label': f'Line style: solid -> shared, dashed -> offline, dotted -> online\nNormalized node/edge frequencies: {online.normalize}\nEdge-annotation position: top -> offline, bottom -> online\nEdge annotation: frequency | avg. duration [days-hours:min]',
                                     'fontname': GV_FONT, 'fontsize': GV_GRAPH_FONTSIZE,
                                     'margin': '0.1,0.1',
                                     'overlap': 'false',
                                     'rankdir': 'LR'})

    # Draw inner nodes
    for act in shared_nodes:
        G.node(act,
               label=f'{act}\n{sum(offline.nodes[act].values()):.2f}\n{sum(online.nodes[act].values()):.2f}',
               shape='box',
               fontname=GV_FONT,
               fontsize=GV_NODE_FONTSIZE,
               style='solid',
               color='black')
        
    for act in offl_nodes:
        G.node(act,
               label=f'{act}\n{sum(offline.nodes[act].values()):.2f}',
               shape='box',
               fontname=GV_FONT,
               fontsize=GV_NODE_FONTSIZE,
               style='dashed',
               color='black')
        
    for act in onl_nodes:
        G.node(act,
               label=f'{act}\n{sum(online.nodes[act].values()):.2f}',
               shape='box',
               fontname=GV_FONT,
               fontsize=GV_NODE_FONTSIZE,
               style='dotted',
               color='black')
    
    # Draw inner arcs
    for ot, (act_a, act_b) in shared_ot_arcs:
        G.edge(act_a, act_b,
               label=f'{offline.arcs[(act_a, act_b)][ot]:.2f} | {td_to_str(offline.arcs_to_avg_dur[(act_a, act_b)][ot])}\n{online.arcs[(act_a, act_b)][ot]:.2f} | {td_to_str(online.arcs_to_avg_dur[(act_a, act_b)][ot])}',
               fontname=GV_FONT,
               fontsize=GV_EDGE_FONTSIZE,
               style='solid',
               color=ot_to_hex_color[ot])
        
    for ot, (act_a, act_b) in offl_ot_arcs:
        G.edge(act_a, act_b,
               label=f'{offline.arcs[(act_a, act_b)][ot]:.2f} | {td_to_str(offline.arcs_to_avg_dur[(act_a, act_b)][ot])}',
               fontname=GV_FONT,
               fontsize=GV_EDGE_FONTSIZE,
               style='dashed',
               color=ot_to_hex_color[ot])
        
    for ot, (act_a, act_b) in onl_ot_arcs:
        G.edge(act_a, act_b,
               label=f'{online.arcs[(act_a, act_b)][ot]:.2f} | {td_to_str(online.arcs_to_avg_dur[(act_a, act_b)][ot])}',
               fontname=GV_FONT,
               fontsize=GV_EDGE_FONTSIZE,
               style='dotted',
               color=ot_to_hex_color[ot])
    
    # Draw source/sink nodes
    for ot, source_sink in shared_ot_source_sinks:
        G.node(source_sink,
               label=f'{source_sink}',
               shape='oval',
               fontname=GV_FONT,
               fontsize=GV_NODE_FONTSIZE,
               style='solid',
               color=ot_to_hex_color[ot])
        
    for ot, source_sink in offl_ot_source_sinks:
        G.node(source_sink,
               label=f'{source_sink}',
               shape='oval',
               fontname=GV_FONT,
               fontsize=GV_NODE_FONTSIZE,
               style='dashed',
               color=ot_to_hex_color[ot])
        
    for ot, source_sink in onl_ot_source_sinks:
        G.node(source_sink,
               label=f'{source_sink}',
               shape='oval',
               fontname=GV_FONT,
               fontsize=GV_NODE_FONTSIZE,
               style='dotted',
               color=ot_to_hex_color[ot])

    # Draw source/sink arcs
    for ot, (x, y) in shared_ot_source_sink_arcs:
        G.edge(x, y,
               fontname=GV_FONT,
               fontsize=GV_EDGE_FONTSIZE,
               style='solid',
               color=ot_to_hex_color[ot])
        
    for ot, (x, y) in offl_ot_source_sink_arcs:
        G.edge(x, y,
               fontname=GV_FONT,
               fontsize=GV_EDGE_FONTSIZE,
               style='dashed',
               color=ot_to_hex_color[ot])
        
    for ot, (x, y) in onl_ot_source_sink_arcs:
        G.edge(x, y,
               fontname=GV_FONT,
               fontsize=GV_EDGE_FONTSIZE,
               style='dotted',
               color=ot_to_hex_color[ot])

    os.makedirs(output_dir, exist_ok=True)
    G.render(filename='tmp', cleanup=True, format='pdf')
    os.replace('tmp.pdf', output_dir / output_file)