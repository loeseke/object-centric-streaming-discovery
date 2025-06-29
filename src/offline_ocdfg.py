"""\
Functionality for translating OC-DFG discovered via pm4py on offline log to components compatible with OcdfgModel.
__author__: "Nina Löseke"
"""

from pm4py import read_ocel2
from pm4py.algo.discovery.ocel.ocdfg.variants.classic import apply as apply_ocel_to_ocdfg
from pm4py.visualization.ocel.ocdfg.variants.classic import apply as apply_ocdfg_to_graphviz, Parameters
from typing import Tuple, Any
import os
from pathlib import Path
import pandas as pd
from statistics import mean


def discover_ocdfg_offline(log_file : str) -> Tuple[dict[str, Any], dict[str, Any]]:
    """
    Discovers OC-DFG and DFGs per object type offline via pm4py on given log and translates components to those used by OcdfgModel.

    Parameters
    ----------
    log_file : str
        Path to log from which OC-DFG and DFGs per object type are discovered via pm4py.

    Returns
    -------
    Tuple[dict[str, Any], dict[str, Any]]
        Dictionary representing merged OC-DFG discovered offline and dictionary representing underlying DFGs per object type.
    """
    # NOTE: no parameters needed; 'edges_performance' in pm4py OC-DFG dict only collates the arc durations in seconds across all occurrences of that arc
    ocel = read_ocel2(log_file)
    ocdfg_dict = apply_ocel_to_ocdfg(ocel)

    # In addition to merged OC-DFG, define nodes/arcs etc. for DFG per object type
    offl_dfgs_per_ot = dict()
    
    # Avoid convergence in activity frequencies
    act_to_ev_ots = dict()
    for ot, act_dict in ocdfg_dict['activities_ot']['events'].items():
        for act, ev_set in act_dict.items():
            act_to_ev_ots.setdefault(act, dict())
            for ev in ev_set:
                act_to_ev_ots[act].setdefault(ev, set())
                act_to_ev_ots[act][ev].add(ot)

    # Sum of activity frequencies per object type by accounting for fraction of OT participation per event
    sum_node_freq = 0.0
    nodes : dict[str, dict[str, float]] = dict()
    for act, ev_dict in act_to_ev_ots.items():
        for _, ot_set in ev_dict.items():
            num_ots = len(ot_set)
            for ot in ot_set:
                nodes.setdefault(act, dict())
                nodes[act].setdefault(ot, 0.0)
                nodes[act][ot] += 1/num_ots
                # Sum up total node frequencies for potential normalization
                sum_node_freq += 1/num_ots  

                # NOTE: not accounting for convergence in offline DFGs, but using absolute frequencies!
                offl_dfgs_per_ot.setdefault(ot, dict())
                offl_dfgs_per_ot[ot].setdefault('nodes', dict())
                offl_dfgs_per_ot[ot]['nodes'].setdefault(act, 0.0)
                offl_dfgs_per_ot[ot]['nodes'][act] += 1 # NOTE: instead of 1/num_ots if adjusting for convergence to avoid wrong frequencies in naively merged OC-DFG
    
    # Translate OC-DFG arcs to format of online model
    sum_arc_freq = 0.0
    ot_arc_to_event_tups = ocdfg_dict['edges']['event_couples']
    arcs : dict[Tuple[str, str], dict[str, float]] = dict()
    for ot, arc_dict in ot_arc_to_event_tups.items():
        for arc, event_tups in arc_dict.items():
            arcs.setdefault(arc, dict())
            arcs[arc].setdefault(ot, 0.0)
            arcs[arc][ot] += len(event_tups)
            sum_arc_freq += len(event_tups)

            offl_dfgs_per_ot.setdefault(ot, dict())
            offl_dfgs_per_ot[ot].setdefault('arcs', dict())
            offl_dfgs_per_ot[ot]['arcs'].setdefault(arc, 0.0)
            offl_dfgs_per_ot[ot]['arcs'][arc] += len(event_tups)
    
    # Translate OC-DFG arc durations to format of online model
    ot_arc_to_durs = ocdfg_dict['edges_performance']['event_couples']
    arcs_to_avg_dur : dict[Tuple[str, str], dict[str, pd.Timedelta]] = dict()
    for ot, arc_dur_dict in ot_arc_to_durs.items():
        for arc, dur_list in arc_dur_dict.items():
            arcs_to_avg_dur.setdefault(arc, dict())
            arcs_to_avg_dur[arc][ot] = pd.Timedelta(seconds=round(mean(dur_list)))

            offl_dfgs_per_ot.setdefault(ot, dict())
            offl_dfgs_per_ot[ot].setdefault('arcs_to_avg_dur', dict())
            offl_dfgs_per_ot[ot]['arcs_to_avg_dur'][arc] = pd.Timedelta(seconds=round(mean(dur_list)))
    
    # Add source-sink nodes and arcs based on start & end activities in OC-DFG dict
    ot_to_initial_acts = ocdfg_dict['start_activities']['events']
    ot_to_final_acts = ocdfg_dict['end_activities']['events']
    source_sink_nodes : dict[str, set[str]] = dict()
    source_sink_arcs : dict[str, set[Tuple[str, str]]] = dict()
    
    for ot, act_dict in ot_to_initial_acts.items():
        source_sink_nodes.setdefault(ot, set())
        source_sink_nodes[ot].add(f'Source {ot}')
        source_sink_arcs.setdefault(ot, set())

        offl_dfgs_per_ot[ot].setdefault('source_sink_nodes', set())
        offl_dfgs_per_ot[ot]['source_sink_nodes'].add(f'Source {ot}')
        offl_dfgs_per_ot[ot].setdefault('source_sink_arcs', set())

        for initial_act in act_dict:
            source_sink_arcs[ot].add((f'Source {ot}', initial_act))
            offl_dfgs_per_ot[ot]['source_sink_arcs'].add((f'Source {ot}', initial_act))

    for ot, act_dict in ot_to_final_acts.items():
        source_sink_nodes.setdefault(ot, set())
        source_sink_nodes[ot].add(f'Sink {ot}')
        source_sink_arcs.setdefault(ot, set())

        offl_dfgs_per_ot[ot].setdefault('source_sink_nodes', set())
        offl_dfgs_per_ot[ot]['source_sink_nodes'].add(f'Sink {ot}')
        offl_dfgs_per_ot[ot].setdefault('source_sink_arcs', set())

        for final_act in act_dict:
            source_sink_arcs[ot].add((final_act, f'Sink {ot}'))
            offl_dfgs_per_ot[ot]['source_sink_arcs'].add((final_act, f'Sink {ot}'))
    
    # Ensure each DFG has necessary attributes, even if empty
    for ot in offl_dfgs_per_ot:
        offl_dfgs_per_ot[ot].setdefault('nodes', dict())
        offl_dfgs_per_ot[ot].setdefault('arcs', dict())
        offl_dfgs_per_ot[ot].setdefault('arcs_to_avg_dur', dict())
        offl_dfgs_per_ot[ot].setdefault('source_sink_nodes', set())
        offl_dfgs_per_ot[ot].setdefault('source_sink_arcs', set())

    # Define entire offline OC-DFG as dict
    offl_ocdfg = {
        'nodes': nodes,
        'arcs': arcs,
        'arcs_to_avg_dur': arcs_to_avg_dur,
        'source_sink_nodes': source_sink_nodes,
        'source_sink_arcs': source_sink_arcs
    }

    return offl_ocdfg, offl_dfgs_per_ot


def visualize_ocdfg_pm4py(log_file : str, edge_thresh : int = 0, act_thresh : int = 0) -> None:
    """
    Draws OC-DFG discovered from log file for given edge- and activity-frequency threshold via pm4py.

    Parameters
    ----------
    log_file : str
        Path to log from which OC-DFG is discovered via pm4py.
    edge_thresh : int, default=0
        Threshold of minimum absolute frequency of arcs in OC-DFG.
    act_thresh : int, default=0
        Threshold of minimum absolute frequency of nodes in OC-DFG.

    Returns
    -------
    None
    """
    ocel = read_ocel2(log_file)
    ocdfg_dict = apply_ocel_to_ocdfg(ocel)

    # Visualize via pm4py and save PDF
    base_dir = Path('pm4py_graphviz_output')
    log_dir = os.path.splitext(os.path.basename(log_file))[0]
    output_parent_dir = base_dir / log_dir
    output_file = f'offline_ocdfg_edge-thresh-{edge_thresh}_act-thresh-{act_thresh}.pdf'

    os.makedirs(output_parent_dir, exist_ok=True)
    G = apply_ocdfg_to_graphviz(ocdfg_dict, parameters={Parameters.EDGE_THRESHOLD: edge_thresh, Parameters.ACT_THRESHOLD: act_thresh})
    G.render('tmp', format='pdf', cleanup=True)
    os.replace('tmp.pdf', output_parent_dir / output_file)