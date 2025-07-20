"""\
Classes and functionality for discovering OCPN from its streaming representation and assessing its accuracy compared to offline model.
__author__: "Nina Löseke"
"""

from typing import Any, Union
import graphviz
import pandas as pd
import os
import numpy as np
import itertools
import uuid
import matplotlib as mpl
import matplotlib.cm as cm
from vars import *
from model_buffers import OcpnBuffer
from cache_policy_buffers import CachePolicy
from priority_policy_buffers import PPBCustom, PPBEventsPerObjectType, PPBLifespanPerObject, PPBLifespanPerObjectType, PPBObjectsPerEvent, PPBObjectsPerObjectType, PPBStridePerObject, PPBStridePerObjectType, PrioPolicyOrder
from utils import EventStream
from pathlib import Path

from model_builder_ocdfg import OcdfgModel
from pm4py.objects.dfg.obj import DFG
from pm4py.algo.discovery.inductive.algorithm import apply as apply_process_tree_to_dfg, Variants as IMVariants
from pm4py.objects.conversion.process_tree.converter import apply as apply_process_tree_to_pn
from pm4py.objects.petri_net.obj import PetriNet
from offline_ocpn import discover_ocpn_offline, visualize_ocpn_pm4py


OCPN_PN = 'petri_nets'
OCPN_ACT = 'activities'
OCPN_DOUBLE_ARCS = 'double_arcs_on_activity'


class OcpnModel(object):
    """
    Represents Object-Centric Petri Net.

    Attributes
    ----------
    double_arc_thresh : float
        Double arc is inserted for an activity and object type if fraction of buffered events where only single object is involved is below threshold.
    model : OcpnBuffer
        Streaming OCPN from which OcpnModel is discovered.
    log : str
        Path to log from which offline OcpnModel is discovered.
    ocpn : dict[str, Any]
        Dictionary containing all components of discovered OCPN.
    ots : set[str]
        Set of all object types that are part of discovered OCPN.
    """
    
    def __init__(self, model_or_log : Union[OcpnBuffer, str], double_arc_thresh : float = 0.5, verbose : bool = False):
        """
        Initializes an OcpnModel object.
        
        Parameters
        ----------
        model_or_log : Union[OcpnBuffer, str]
            Streaming OCPN or path to offline log from which OcpnModel is discovered.
        double_arc_thresh : float, default=0.5
            Threshold for inserting single vs. double arcs.
        verbose : bool, default=False
            If enabled, discovered OCPN is printed to terminal.
        
        Raises
        ------
        RuntimeError
            If neither streaming representation nor log to discover OCPN from are specified correctly.
        """
        self.double_arc_thresh = double_arc_thresh
        self.model : OcpnBuffer = None
        self.log : str = None
        self.ocpn : dict[str, Any] = {OCPN_PN: dict(), OCPN_ACT: set(), OCPN_DOUBLE_ARCS: dict()}
        self.ots : set[str] = set()

        if isinstance(model_or_log, OcpnBuffer):
            self.model = model_or_log
            self.ocdfg_model = OcdfgModel(self.model.ocdfg_buf)
            self.__create_pn_per_ot()
            self.__create_double_arcs()
            self.ots = set(self.ocdfg_model.dfgs_per_ot.keys())
        elif isinstance(model_or_log, str):
            self.log = model_or_log
            self.ocpn = discover_ocpn_offline(model_or_log, double_arc_thresh)
            self.ots = set(self.ocpn[OCPN_PN].keys())
        else:
            raise RuntimeError(f'Cannot build OcpnModel from input of type {type(model_or_log)}.') 

        if verbose:
            print(self)
    
    def __create_double_arcs(self) -> None:
        """Determine which arcs entering/leaving transitions for a certain activity and object type should be double according to event-activity buffer."""
        self.ocpn[OCPN_DOUBLE_ARCS] = {ot: dict() for ot in self.model.ea_buf.get_buffered_nested_values_for_key(OBJECT_TYPE)}
        # Adapted from pm4py; source: https://github.com/process-intelligence-solutions/pm4py/blob/release/pm4py/algo/discovery/ocel/ocpn/variants/classic.py#L130
        for act_key, act_dicts in self.model.ea_buf.buf.items():
            event_count = dict()
            single_obj_count = dict()
            for act_dict in act_dicts:
                ot = act_dict[OBJECT_TYPE]

                event_count.setdefault(ot, 0)
                event_count[ot] += 1

                single_obj_count.setdefault(ot, 0)
                if act_dict[HAS_SINGLE_OBJ]:
                    single_obj_count[ot] += 1

            for ot in single_obj_count:
                self.ocpn[OCPN_DOUBLE_ARCS][ot][act_key] = single_obj_count[ot] / event_count[ot] <= self.double_arc_thresh

    def __create_pn_per_ot(self) -> None:
        """Discover Petri nets per object type from OcdfgPerObjectType."""
        for ot, dfg_model in self.ocdfg_model.dfgs_per_ot.items():
            start_activities = list()
            end_activities = list()
            for arc_src, arc_target in dfg_model.source_sink_arcs:
                if arc_src == f'Source {ot}':
                    start_activities.append(arc_target)
                else:
                    end_activities.append(arc_src)

            if len(dfg_model.arcs) > 0:
                dfg = DFG(graph=dfg_model.arcs, start_activities=start_activities, end_activities=end_activities)
            else:
                # Build pm4py purely from start and end activities if buffered DFG lacks "inner" arcs
                dfg = DFG(start_activities=start_activities, end_activities=end_activities)

            process_tree = apply_process_tree_to_dfg(
                dfg, 
                variant=IMVariants.IMd, 
                # NOTE: fall-throughs and strict sequence cuts disabled for performance reasons
                # Source: https://github.com/process-intelligence-solutions/pm4py/blob/release/pm4py/algo/discovery/ocel/ocpn/variants/classic.py#L139-L144
                parameters={
                    "disable_fallthroughs": True, 
                    "disable_strict_sequence_cut": True
                }
            )
            petri_net, initial_marking, final_marking = apply_process_tree_to_pn(process_tree)
            self.ocpn[OCPN_PN][ot] = (petri_net, initial_marking, final_marking)
            
            # Separately save activity nodes of DFGs
            self.ocpn[OCPN_ACT].update(dfg_model.nodes.keys())

    def get_graphviz_file_name(self) -> str:
        """
        Derives unique file name for visualization output of OCPN that encodes its discovery parameters, e.g. coupled removal.
        
        Returns
        -------
        str
        """
        if self.model is not None:
            return f"cr-{int(self.model.coupled_removal)}.pdf"
        else:
            return f"offline.pdf"
    
    def get_graphviz_subdir(self) -> Path:
        """
        Derives unique path name for visualization output of OCPN that encodes its streaming parameters, e.g. event-activity buffer size, cache policy.
        
        Returns
        -------
        Path
            Subdirectory to save graphviz output to. If OCPN is discovered offline, "ocpn" subdirectory is used.
        """
        if self.model is not None:
            pp_name = f'{self.model.pp_buf.prio_order.value.lower()}-{self.model.pp_buf.pp.value.lower().replace(' ', '-')}' if self.model.pp_buf is not None else 'none'
            pp_dfgs_name = f'{self.model.pp_buf_dfgs.prio_order.value.lower()}-{self.model.pp_buf_dfgs.pp.value.lower().replace(' ', '-')}' if self.model.pp_buf_dfgs is not None else 'none'

            return Path("ocpn",
                        f"{self.model.node_buf_size}_{self.model.arc_buf_size}_{self.model.ea_buf_size}",
                        f"{self.model.cp.value.lower()}",
                        f"ea-{pp_name}_dfg-{pp_dfgs_name}")
        else:
            return Path("ocpn")

    def visualize(self, output_dir : Path, output_file : str, ot_to_hex_color : dict[str, Any]) -> None:
        """
        Draws discovered, merged OCPN via graphviz. Object types are highlighted in the given colors.
        Given output directory is created automatically if it does not yet exist.

        Parameters
        ----------
        output_dir : Path
            Output directory to which PDF is saved.
        output_file : str
            Name of file to which PDF is saved.
        ot_to_hex_color : dict[str, Any]
            Mapping of object types in the OCPN to hex color codes.

        Returns
        -------
        None
        """
        # Adapted from pm4py; source: https://github.com/process-intelligence-solutions/pm4py/blob/release/pm4py/visualization/ocel/ocpn/variants/wo_decoration.py
        G = graphviz.Digraph(
            graph_attr={'label': f'Line style: bold->variable arc, normal->non-variable arc',
                        'fontname': GV_FONT, 
                        'fontsize': GV_GRAPH_FONTSIZE,
                        'margin': '0.1,0.1',
                        'overlap': 'false',
                        'rankdir': 'LR'},
            engine="dot",
        )
        G.attr("node", shape="ellipse", fixedsize="false", fontname=GV_FONT, fontsize=GV_NODE_FONTSIZE)

        activities_map = {}
        transition_map = {}
        places = {}

        for act in self.ocpn[OCPN_ACT]:
            activities_map[act] = str(uuid.uuid4())
            G.node(activities_map[act], label=act, shape="box")

        for ot in self.ocpn[OCPN_PN]:
            otc = ot_to_hex_color[ot]
            net, im, fm = self.ocpn[OCPN_PN][ot]

            for place in net.places:
                place_id = str(uuid.uuid4())
                places[place] = place_id
                place_label = " "
                place_shape = "circle"
                place_color = "black"
                place_fillcolor = otc

                # Insert source/sink nodes
                if place in im:
                    place_label = f'Source {ot}'
                    place_shape = "ellipse"
                    place_color = otc
                    place_fillcolor = None
                elif place in fm:
                    place_label = f'Sink {ot}'
                    place_shape = "ellipse"
                    place_color = otc
                    place_fillcolor = None

                G.node(
                    places[place],
                    label=place_label,
                    shape=place_shape,
                    style="filled" if place_fillcolor is not None else None,
                    fontcolor='black',
                    color=place_color,
                    fillcolor=place_fillcolor
                )

            for trans in net.transitions:
                if trans.label is not None:
                    transition_map[trans] = activities_map[trans.label]
                else:
                    transition_map[trans] = str(uuid.uuid4())
                    G.node(
                        transition_map[trans],
                        label=" ",
                        shape="box",
                        style="filled",
                        fillcolor=otc,
                    )

            for arc in net.arcs:
                arc_label = " "
                if type(arc.source) is PetriNet.Place:
                    is_double = (
                        arc.target.label in self.ocpn[OCPN_DOUBLE_ARCS][ot]
                        and self.ocpn[OCPN_DOUBLE_ARCS][ot][arc.target.label]
                    )
                    penwidth = "4.0" if is_double else "1.0"
                    G.edge(
                        places[arc.source],
                        transition_map[arc.target],
                        color=otc,
                        penwidth=penwidth,
                        label=arc_label,
                    )
                elif type(arc.source) is PetriNet.Transition:
                    is_double = (
                        arc.source.label in self.ocpn[OCPN_DOUBLE_ARCS][ot]
                        and self.ocpn[OCPN_DOUBLE_ARCS][ot][arc.source.label]
                    )
                    penwidth = "4.0" if is_double else "1.0"
                    G.edge(
                        transition_map[arc.source],
                        places[arc.target],
                        color=otc,
                        penwidth=penwidth,
                        label=arc_label,
                    )

        os.makedirs(output_dir, exist_ok=True)
        G.render(filename='tmp', cleanup=True, format='pdf')
        os.replace('tmp.pdf', output_dir / output_file)
    
    def __str__(self) -> str:
        """Creates string listing all OCPN components."""
        ret = 'Merged OCPN:'
        ret += f'\nObject types:'
        for i, ot in enumerate(self.ocpn['petri_nets']):
            ret += f'\n{i+1}. {ot}'

        ret += f'\nActivities:'
        for i, act in enumerate(self.ocpn[OCPN_ACT]):
            ret += f'\n{i+1}. {act}'
            for ot in self.ocpn['petri_nets']:
                if act in self.ocpn[OCPN_DOUBLE_ARCS][ot]:
                    if self.ocpn[OCPN_DOUBLE_ARCS][ot][act]:
                        ret += f'\n * double for OT {ot}'
                    else:
                        ret += f'\n * single for OT {ot}'
        
        return ret


def get_ocpn_accuracy(offline : OcpnModel, online : OcpnModel) -> dict[str, float]:
    """
    Computes evaluation metrics to assess quality of online model compared to offline model, i.e. accuracy, precision, and recall for double arcs and activity-place-activity pairs.

    Parameters
    ----------
    offline : OcpnModel
        OCPN discovered offline from full log.
    online : OcpnModel
        OCPN discovered from streaming representation.
    
    Returns
    -------
    dict[str, float]
        Mapping of evaluation metrics to values.
    """
    # Compute accuracy & precision similar to structural evaluation of Petri nets in S-BAR paper
    # I.e. compare which activities are connected via a place
    offl_pos_act_pairs = set()
    offl_neg_act_pairs = set()
    onl_pos_act_pairs = set()
    offl_pos_act_pairs = set()
    possible_act_pairs = set()

    for ocpn, is_online in [(offline.ocpn, False), (online.ocpn, True)]:
        for ot, pn_tup in ocpn[OCPN_PN].items():
            pn, _, _ = pn_tup
            ot_activities = set()
            for place in pn.places:
                for place_in_arc in place.in_arcs:
                    for place_out_arc in place.out_arcs:
                        place_pred = place_in_arc.source.label
                        place_succ = place_out_arc.target.label
                        ot_activities.update([place_pred, place_succ])

                        if place_pred is None or place_succ is None:
                            continue
                        else:
                            if not is_online:
                                offl_pos_act_pairs.add((ot, place_pred, place_succ))
                            else:
                                onl_pos_act_pairs.add((ot, place_pred, place_succ))
            
            ot_activities.discard(None)
            ot_possible_act_pairs = itertools.product(ot_activities, ot_activities)
            possible_act_pairs.update([(ot, a, b) for a, b in ot_possible_act_pairs])

    onl_neg_act_pairs = possible_act_pairs - onl_pos_act_pairs
    offl_neg_act_pairs = possible_act_pairs - offl_pos_act_pairs
    
    act_pair_fp = len(onl_pos_act_pairs - offl_pos_act_pairs)
    act_pair_tp = len(onl_pos_act_pairs.intersection(offl_pos_act_pairs))
    act_pair_fn = len(onl_neg_act_pairs - offl_neg_act_pairs)
    act_pair_tn = len(onl_neg_act_pairs.intersection(offl_neg_act_pairs))

    act_pair_acc = (act_pair_tp + act_pair_tn) / (act_pair_tp + act_pair_tn + act_pair_fp + act_pair_fn) if act_pair_tp + act_pair_tn + act_pair_fp + act_pair_fn > 0 else None
    act_pair_prec = act_pair_tp / (act_pair_tp + act_pair_fp) if act_pair_tp + act_pair_fp > 0 else None
    act_pair_rec = act_pair_tp / (act_pair_tp + act_pair_fn) if act_pair_tp + act_pair_fn > 0 else None

    # Compute double-arc accuracy & precision based on whether activities require variable in-/out-arcs or not
    offl_pos_double_arcs = set()
    offl_neg_double_arcs = set()
    onl_pos_double_arcs = set()
    onl_neg_double_arcs = set()

    for ocpn, is_online in [(offline.ocpn, False), (online.ocpn, True)]:
        for ot, act_dict in ocpn[OCPN_DOUBLE_ARCS].items():
            for act, is_double_act in act_dict.items():
                if is_double_act:
                    if is_online:
                        onl_pos_double_arcs.add((ot, act))
                    else:
                        offl_pos_double_arcs.add((ot, act))
                else:
                    if is_online:
                        onl_neg_double_arcs.add((ot, act))
                    else:
                        offl_neg_double_arcs.add((ot, act))
    
    double_arc_tp = len(onl_pos_double_arcs.intersection(offl_pos_double_arcs))
    double_arc_fp = len(onl_pos_double_arcs - offl_pos_double_arcs)
    double_arc_tn = len(onl_neg_double_arcs.intersection(offl_neg_double_arcs))
    double_arc_fn = len(onl_neg_double_arcs - offl_neg_double_arcs)

    double_arc_acc = (double_arc_tp + double_arc_tn) / (double_arc_tp + double_arc_tn + double_arc_fp + double_arc_fn) if double_arc_tp + double_arc_tn + double_arc_fp + double_arc_fn > 0 else None
    double_arc_prec = double_arc_tp / (double_arc_tp + double_arc_fp) if double_arc_tp + double_arc_fp > 0 else None
    double_arc_rec = double_arc_tp / (double_arc_tp + double_arc_fn) if double_arc_tp + double_arc_fn > 0 else None

    # NOTE: source-sink arc accuracy/precision not feasible since their may be indistinguishable tau transitions

    return {'act-place-act recall': act_pair_rec,
            'act-place-act accuracy': act_pair_acc,
            'act-place-act precision': act_pair_prec,
            'double-arc recall': double_arc_rec,
            'double-arc accuracy': double_arc_acc,
            'double-arc precision': double_arc_prec}


def get_ocpn_avg_scores(offline : OcpnModel, online : OcpnModel) -> dict[str, float]:
    """
    Averages OCPN accuracy, precision, and recall across double arcs and activity-place-activity pairs.

    Parameters
    ----------
    offline : OcpnModel
        OCPN discovered offline from full log.
    online : OcpnModel
        OCPN discovered from streaming representation.

    Returns
    -------
    dict[str, float]
        Mapping of evaluation metrics to values.
    """
    scores = get_ocpn_accuracy(offline, online)
    rec_scores = [scores['act-place-act recall'], scores['double-arc recall']]
    prec_scores = [scores['act-place-act precision'], scores['double-arc precision']]
    acc_scores = [scores['act-place-act accuracy'], scores['double-arc accuracy']]
    filtered_rec_scores = [rec for rec in rec_scores if rec is not None]
    filtered_prec_scores = [prec for prec in prec_scores if prec is not None]
    filtered_acc_scores = [acc for acc in acc_scores if acc is not None]

    return {'recall': np.mean(filtered_rec_scores) if len(filtered_rec_scores) > 0 else None,
            'precision': np.mean(filtered_prec_scores) if len(filtered_prec_scores) > 0 else None,
            'accuracy': np.mean(filtered_acc_scores) if len(filtered_acc_scores) > 0 else None}