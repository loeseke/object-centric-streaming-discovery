"""\
Functionality for translating OCPN discovered via pm4py on offline log to components compatible with OcpnModel.
__author__: "Nina Löseke"
"""

from pm4py import read_ocel2
from pm4py.visualization.petri_net.visualizer import apply as apply_pn_to_graphviz
from pm4py.algo.discovery.ocel.ocpn.algorithm import apply as apply_ocel_to_ocpn
from pm4py.visualization.ocel.ocpn.visualizer import apply as apply_ocpn_to_graphviz
from pm4py.algo.discovery.ocel.ocpn.variants.classic import Parameters
from typing import Any
import os
from pathlib import Path


def discover_ocpn_offline(log_file : str, double_arc_thresh : float = 0.5) -> dict[str, Any]:
    """
    Parameters
    ----------
    log_file : str
        Path to log from which OCPN is discovered via pm4py.
    double_arc_thresh : float, default=0.5
        Double arc is inserted for an activity and object type if fraction of events where only single object is involved is below threshold.

    Returns
    -------
    dict[str, Any]
    """
    ocel = read_ocel2(log_file)
    # NOTE: per default, applies Inductive Miner on flattened OCEL per OT
    # Returns dict w/ keys "double_arcs_on_activity" and "petri_nets", among others
    ocpn = apply_ocel_to_ocpn(ocel, parameters={Parameters.DOUBLE_ARC_THRESHOLD: double_arc_thresh})
    return ocpn


def visualize_ocpn_pm4py(log_file : str, double_arc_thresh : float = 0.5, visualize_pns : bool = False) -> None:
    """
    Draws OCPN discovered from log file for given double-arc threshold via pm4py.

    Parameters
    ----------
    log_file : str
        Path to log from which OCPN is discovered via pm4py. 
    double_arc_thresh : float, default=0.5
        Double arc is inserted for an activity and object type if fraction of events where only single object is involved is below threshold.
    visualize_pns : bool, default=False
        If set to True, individual Petri nets per object type are also drawn.

    Returns
    -------
    None
    """
    ocel = read_ocel2(log_file)
    ocpn = apply_ocel_to_ocpn(ocel)
    
    # Visualize via pm4py and save PDF
    base_dir = Path('pm4py_graphviz_output')
    log_dir = os.path.splitext(os.path.basename(log_file))[0]
    output_parent_dir = base_dir / log_dir
    output_file = f'offline_ocpn_double-arc-thresh-{double_arc_thresh}.pdf'

    os.makedirs(output_parent_dir, exist_ok=True)
    G = apply_ocpn_to_graphviz(ocpn, parameters={Parameters.DOUBLE_ARC_THRESHOLD: double_arc_thresh})
    G.render(filename='tmp', cleanup=True, format='pdf')
    os.replace('tmp.pdf', output_parent_dir / output_file)

    if visualize_pns:
        for ot, pn_tup in ocpn['petri_nets'].items():
            output_file = f'offline_pn_ot-{ot.replace(' ', '-').lower()}.pdf'
            pn, initial_m, final_m = pn_tup
            G_pn = apply_pn_to_graphviz(pn, initial_marking=initial_m, final_marking=final_m)
            G_pn.render('tmp', format='pdf', cleanup=True)
            os.replace('tmp.pdf', output_parent_dir / output_file)