"""\
Functionality for translating TOTeM model discovered via reference implementation on offline log to TotemModel. This file makes some slight changes to the reference implementation available at https://github.com/LukasLiss/TOTeM-temporal-object-type-model/blob/main/algorithm.py.
__author__: "Lukas Liss, Nina Löseke"
"""

from datetime import datetime
from typing import Any, Tuple, Union
from ocpa.objects.log.importer.ocel2.xml import factory as ocel_import_factory
from vars import *
from typing import Any

# NOTE: need to install OCPA 1.3.3 from Github repository; automatically downgrades pm4py to 2.2.32 as requirement (whereas up to 2.7.15 currently available)

# Temporal relation constants (constants serving as a representation that is easier to understand than just the numbers)
TR_TOTAL = "total"

# Event cardinality constants
EC_TOTAL = "total"

# Log cardinality constants
LC_TOTAL = "total"


def get_all_event_objects(ocel : Any, event_id : str) -> list[str]:
    """
    Extracts list of objects participating in given event in given log.

    Parameters
    ----------
    ocel : Any
        OCEL 2.0 log imported via ocpa.
    event_id : str
        Unique identifier of event in log.

    Returns
    -------
    list[str]
    """
    obj_ids = []
    for obj_type in ocel.object_types:
        obj_ids += ocel.get_value(event_id, obj_type)
    return obj_ids


def get_most_precise_lc(directed_type_tuple : Tuple[str, str], tau : float, log_cardinalities : dict[Tuple[str, str], dict[str, int]]) -> Union[None, str]:
    """
    Determines most precise log cardinality for given directed type pair and filter parameter tau.

    Parameters
    ----------
    directed_type_tuple : Tuple[str, str]
        Directed pair of object types.
    tau : float
        Filter parameter tau in [0, 1]
    log_cardinalities : dict[Tuple[str, str], dict[str, int]]
        Mapping of directed type pairs to dictionaries containing counts of how often each log cardinality holds.
    
    Returns
    -------
    Union[None, str]
        String of most precise log cardinality or None, if no log cardinality can be determined for given type pair.
    """
    total = 0
    if directed_type_tuple in log_cardinalities.keys() and LC_TOTAL in log_cardinalities[directed_type_tuple].keys():
        total = log_cardinalities[directed_type_tuple][LC_TOTAL]

    if total == 0:
        return None

    if (LC_ONE in log_cardinalities[directed_type_tuple].keys()) and (
            (log_cardinalities[directed_type_tuple][LC_ONE] / total) >= tau):
        return LC_ONE
    
    if (LC_ZERO_ONE in log_cardinalities[directed_type_tuple].keys()) and (
            (log_cardinalities[directed_type_tuple][LC_ZERO_ONE] / total) >= tau):
        return LC_ZERO_ONE
    
    if (LC_ONE_MANY in log_cardinalities[directed_type_tuple].keys()) and (
            (log_cardinalities[directed_type_tuple][LC_ONE_MANY] / total) >= tau):
        return LC_ONE_MANY
    
    if (LC_ZERO_MANY in log_cardinalities[directed_type_tuple].keys()) and (
            (log_cardinalities[directed_type_tuple][LC_ZERO_MANY] / total) >= tau):
        return LC_ZERO_MANY

    return None


def get_most_precise_ec(directed_type_tuple : Tuple[str, str], tau : float, event_cardinalities : dict[Tuple[str, str], dict[str, int]]) -> Union[None, str]:
    """
    Determines most precise event cardinality for given directed type pair and filter parameter tau.

    Parameters
    ----------
    directed_type_tuple : Tuple[str, str]
        Directed pair of object types.
    tau : float
        Filter parameter tau in [0, 1]
    event_cardinalities : dict[Tuple[str, str], dict[str, int]]
        Mapping of directed type pairs to dictionaries containing counts of how often each event cardinality holds.
    
    Returns
    -------
    Union[None, str]
        String of most precise event cardinality or None, if no event cardinality can be determined for given type pair.
    """
    total = 0
    if directed_type_tuple in event_cardinalities.keys() and EC_TOTAL in event_cardinalities[
        directed_type_tuple].keys():
        total = event_cardinalities[directed_type_tuple][EC_TOTAL]

    if total == 0:
        return None

    if (EC_ZERO in event_cardinalities[directed_type_tuple].keys()) and (
            (event_cardinalities[directed_type_tuple][EC_ZERO] / total) >= tau):
        return EC_ZERO
    
    if (EC_ONE in event_cardinalities[directed_type_tuple].keys()) and (
            (event_cardinalities[directed_type_tuple][EC_ONE] / total) >= tau):
        return EC_ONE
    
    if (EC_ZERO_ONE in event_cardinalities[directed_type_tuple].keys()) and (
            (event_cardinalities[directed_type_tuple][EC_ZERO_ONE] / total) >= tau):
        return EC_ZERO_ONE
    
    if (EC_ONE_MANY in event_cardinalities[directed_type_tuple].keys()) and (
            (event_cardinalities[directed_type_tuple][EC_ONE_MANY] / total) >= tau):
        return EC_ONE_MANY
    
    if (EC_ZERO_MANY in event_cardinalities[directed_type_tuple].keys()) and (
            (event_cardinalities[directed_type_tuple][EC_ZERO_MANY] / total) >= tau):
        return EC_ZERO_MANY

    return None


def get_most_precise_tr(directed_type_tuple : Tuple[str, str], tau : float, temporal_relation : dict[Tuple[str, str], dict[str, int]]) -> Union[None, str]:
    """
    Determines most precise temporal relation for given directed type pair and filter parameter tau.

    Parameters
    ----------
    directed_type_tuple : Tuple[str, str]
        Directed pair of object types.
    tau : float
        Filter parameter tau in [0, 1]
    temporal_relation : dict[Tuple[str, str], dict[str, int]]
        Mapping of directed type pairs to dictionaries containing counts of how often each temporal relation holds between them.
    
    Returns
    -------
    Union[None, str]
        String of most precise temporal relation or None, if no temporal relation can be determined for given type pair.
    """
    total = 0
    if directed_type_tuple in temporal_relation.keys() and TR_TOTAL in temporal_relation[directed_type_tuple].keys():
        total = temporal_relation[directed_type_tuple][TR_TOTAL]

    if total == 0:
        return None

    if (TR_DURING in temporal_relation[directed_type_tuple].keys()) and (
            (temporal_relation[directed_type_tuple][TR_DURING] / total) >= tau):
        return TR_DURING
    
    if (TR_DURING_INVERSE in temporal_relation[directed_type_tuple].keys()) and (
            (temporal_relation[directed_type_tuple][TR_DURING_INVERSE] / total) >= tau):
        return TR_DURING_INVERSE
    
    if (TR_PRECEDES in temporal_relation[directed_type_tuple].keys()) and (
            (temporal_relation[directed_type_tuple][TR_PRECEDES] / total) >= tau):
        return TR_PRECEDES
    
    if (TR_PRECEDES_INVERSE in temporal_relation[directed_type_tuple].keys()) and (
            (temporal_relation[directed_type_tuple][TR_PRECEDES_INVERSE] / total) >= tau):
        return TR_PRECEDES_INVERSE
    
    if (TR_PARALLEL in temporal_relation[directed_type_tuple].keys()) and (
            (temporal_relation[directed_type_tuple][TR_PARALLEL] / total) >= tau):
        return TR_PARALLEL

    return None


def discover_totem_offline(
        file_path : str, 
        tau : float = 1, 
        verbose : bool = False
    ) -> Tuple[set[str], dict[Tuple[str, str], dict[str, Any]]]:
    """
    Mines TOTeM model from given offline log for given threshold parameter tau.
    
    Parameters
    ----------
    file_path : str
        Path of OCEl 2.0 log from which TOTeM model is discovered.
    tau : float, default=1
        Filter parameter in [0, 1] functioning as minimum threshold for determining most precise relations.
    verbose : bool, default=False
        If True, resulting TOTeM model is printed to terminal.
    
    Returns
    -------
    Tuple[set[str], dict[str, dict[str, Any]]]
        Nodes corresponding to object types and mapping of arcs to corresponding temporal, log-cardinality, and event-cardinality relation.
    """
    assert 0 <= tau <= 1
    ocel = ocel_import_factory.apply(file_path)

    # temporal relations results
    temporal_rels: dict[tuple[str, str], dict[str, int]] = dict()  # stores all the temporal relations found
    # event cardinality results
    event_cards: dict[tuple[str, str], dict[str, int]] = dict()  # stores all the temporal cardinalities found
    # event cardinality results
    log_cards: dict[tuple[str, str], dict[str, int]] = dict()  # stores all the temporal cardinalities found

    # object min times 
    # str identifier of the object maps to the earliest time recorded for that object in the event log
    o_min_times: dict[str, datetime] = dict()

    # object max times
    # str identifier of the object maps to the last time recorded for that object in the event log
    o_max_times: dict[str, datetime] = dict()

    # get a list of all object types (or variable that is filled while passing through the process executions)
    type_relations: set[set[str, str]] = set()  # stores all connected types

    o2o: dict[str, dict[str, set[str]]] = dict()

    # a mapping from type to its objects
    type_to_object = dict()

    for px in ocel.process_executions:
        for ev in px:
            # event infos: objects and timestamps
            # NOTE: sufficient for AgeOfEmpires10Matches and ContainerLogistics log; assumes datetime.isoformat
            ev_timestamp = ocel.get_value(ev, 'event_timestamp')

            objects_of_event = get_all_event_objects(ocel, ev)
            for obj in objects_of_event:
                # o2o updating
                o2o.setdefault(obj, dict())
                for type in ocel.object_types:
                    o2o[obj].setdefault(type, set())
                    o2o[obj][type].update(
                        ocel.get_value(ev, type))  # add all objects connected via e2o to each object involved
                
                # update lifespan information
                o_min_times.setdefault(obj, ev_timestamp)
                if ev_timestamp < o_min_times[obj]:
                    o_min_times[obj] = ev_timestamp
                
                o_max_times.setdefault(obj, ev_timestamp)
                if ev_timestamp > o_max_times[obj]:
                    o_max_times[obj] = ev_timestamp

            # compute event cardinality
            involved_types = []
            obj_count_per_type = dict()
            for type in ocel.object_types:
                obj_list = ocel.get_value(ev, type)
                
                if not obj_list:
                    continue
                else:
                    type_to_object.setdefault(type, set())
                    type_to_object[type].update(obj_list)
                    involved_types.append(type)
                    obj_count_per_type[type] = len(obj_list)
            
            # create related types
            for t1 in involved_types:
                for t2 in involved_types:
                    if t1 != t2:
                        type_relations.add(frozenset({t1, t2}))
            
            # for all type pairs determine
            for type_source in involved_types:
                for type_target in ocel.object_types:
                    # add one to total
                    event_cards.setdefault((type_source, type_target), dict())
                    event_cards[(type_source, type_target)].setdefault(EC_TOTAL, 0)
                    event_cards[(type_source, type_target)][EC_TOTAL] += 1
                    
                    # determine cardinality
                    cardinality = 0
                    if type_target in obj_count_per_type.keys():
                        cardinality = obj_count_per_type[type_target]
                    
                    # add one to matching cardinalities
                    if cardinality == 0:
                        event_cards[(type_source, type_target)].setdefault(EC_ZERO, 0)
                        event_cards[(type_source, type_target)][EC_ZERO] += 1
                        event_cards[(type_source, type_target)].setdefault(EC_ZERO_ONE, 0)
                        event_cards[(type_source, type_target)][EC_ZERO_ONE] += 1
                        event_cards[(type_source, type_target)].setdefault(EC_ZERO_MANY, 0)
                        event_cards[(type_source, type_target)][EC_ZERO_MANY] += 1
                    
                    elif cardinality == 1:
                        event_cards[(type_source, type_target)].setdefault(EC_ONE, 0)
                        event_cards[(type_source, type_target)][EC_ONE] += 1
                        event_cards[(type_source, type_target)].setdefault(EC_ZERO_ONE, 0)
                        event_cards[(type_source, type_target)][EC_ZERO_ONE] += 1
                        event_cards[(type_source, type_target)].setdefault(EC_ONE_MANY, 0)
                        event_cards[(type_source, type_target)][EC_ONE_MANY] += 1
                        event_cards[(type_source, type_target)].setdefault(EC_ZERO_MANY, 0)
                        event_cards[(type_source, type_target)][EC_ZERO_MANY] += 1
                    
                    elif cardinality > 1:
                        event_cards[(type_source, type_target)].setdefault(EC_ONE_MANY, 0)
                        event_cards[(type_source, type_target)][EC_ONE_MANY] += 1
                        event_cards[(type_source, type_target)].setdefault(EC_ZERO_MANY, 0)
                        event_cards[(type_source, type_target)][EC_ZERO_MANY] += 1

    # merge o2o and e2o connected objects
    for (source_o, target_o) in ocel.o2o_graph.graph.edges:
        # print(f"{source_o} - {target_o}")
        type_of_target_o = None
        type_of_source_o = None
        
        for type in ocel.object_types:
            if target_o in type_to_object[type]:
                type_of_target_o = type
                break
        
        for type in ocel.object_types:
            if source_o in type_to_object[type]:
                type_of_source_o = type
                break
        
        if type_of_target_o == None or type_of_source_o == None or type_of_target_o == type_of_source_o:
            continue
        
        # add O2O-related objects to O2Os
        # make sure each potentially new object is mapped to all types and objects of respective types involved in O2O
        o2o.setdefault(source_o, dict())
        for type in ocel.object_types:
            o2o[source_o].setdefault(type, set())
        o2o[source_o][type_of_target_o].add(target_o)
        
        o2o.setdefault(target_o, dict())
        for type in ocel.object_types:
            o2o[target_o].setdefault(type, set())
        o2o[target_o][type_of_source_o].add(source_o)

        # add O2O-related OTs to connected types
        type_relations.add(frozenset({type_of_source_o, type_of_target_o}))

    # compute log cardinality
    for type_source in ocel.object_types:
        for type_target in ocel.object_types:
            temporal_rels.setdefault((type_source, type_target), dict())
            
            for obj in type_to_object[type_source]:
                log_cards.setdefault((type_source, type_target), dict())
                log_cards[(type_source, type_target)].setdefault(LC_TOTAL, 0)
                log_cards[(type_source, type_target)][LC_TOTAL] += 1

                cardinality = len(o2o[obj][type_target])

                if cardinality == 0:
                    log_cards[(type_source, type_target)].setdefault(LC_ZERO_ONE, 0)
                    log_cards[(type_source, type_target)][LC_ZERO_ONE] += 1
                    log_cards[(type_source, type_target)].setdefault(LC_ZERO_MANY, 0)
                    log_cards[(type_source, type_target)][LC_ZERO_MANY] += 1
                
                elif cardinality == 1:
                    log_cards[(type_source, type_target)].setdefault(LC_ONE, 0)
                    log_cards[(type_source, type_target)][LC_ONE] += 1
                    log_cards[(type_source, type_target)].setdefault(LC_ZERO_ONE, 0)
                    log_cards[(type_source, type_target)][LC_ZERO_ONE] += 1
                    log_cards[(type_source, type_target)].setdefault(LC_ONE_MANY, 0)
                    log_cards[(type_source, type_target)][LC_ONE_MANY] += 1
                    log_cards[(type_source, type_target)].setdefault(LC_ZERO_MANY, 0)
                    log_cards[(type_source, type_target)][LC_ZERO_MANY] += 1
                
                elif cardinality > 1:
                    log_cards[(type_source, type_target)].setdefault(LC_ONE_MANY, 0)
                    log_cards[(type_source, type_target)][LC_ONE_MANY] += 1
                    log_cards[(type_source, type_target)].setdefault(LC_ZERO_MANY, 0)
                    log_cards[(type_source, type_target)][LC_ZERO_MANY] += 1

                # compute temporal relations
                for obj_target in o2o[obj][type_target]:
                    temporal_rels[(type_source, type_target)].setdefault(TR_TOTAL, 0)
                    temporal_rels[(type_source, type_target)][TR_TOTAL] += 1
                    
                    if o_min_times[obj_target] <= o_min_times[obj] <= o_max_times[obj] <= o_max_times[obj_target]:
                        temporal_rels[(type_source, type_target)].setdefault(TR_DURING, 0)
                        temporal_rels[(type_source, type_target)][TR_DURING] += 1
                    
                    if o_min_times[obj] <= o_min_times[obj_target] <= o_max_times[obj_target] <= o_max_times[obj]:
                        temporal_rels[(type_source, type_target)].setdefault(TR_DURING_INVERSE, 0)
                        temporal_rels[(type_source, type_target)][TR_DURING_INVERSE] += 1
                    
                    if (o_min_times[obj] <= o_max_times[obj] <= o_min_times[obj_target] <= o_max_times[obj_target]) or (
                            o_min_times[obj] < o_min_times[obj_target] <= o_max_times[obj] < o_max_times[obj_target]):
                        temporal_rels[(type_source, type_target)].setdefault(TR_PRECEDES, 0)
                        temporal_rels[(type_source, type_target)][TR_PRECEDES] += 1
                    
                    if (o_min_times[obj_target] <= o_max_times[obj_target] <= o_min_times[obj] <= o_max_times[obj]) or (
                            o_min_times[obj_target] < o_min_times[obj] <= o_max_times[obj_target] < o_max_times[obj]):
                        temporal_rels[(type_source, type_target)].setdefault(TR_PRECEDES_INVERSE, 0)
                        temporal_rels[(type_source, type_target)][TR_PRECEDES_INVERSE] += 1
                    
                    # allways parallel
                    temporal_rels[(type_source, type_target)].setdefault(TR_PARALLEL, 0)
                    temporal_rels[(type_source, type_target)][TR_PARALLEL] += 1

    # define nodes and arcs according to TOTeM model builder
    nodes : set[str] = set()
    edges : dict[str, dict[str, Any]] = dict()

    # for each connection give the 6 relations
    for connected_types in type_relations:
        # define model nodes
        t1, t2 = connected_types
        nodes.add(t1)
        nodes.add(t2)

        # define model arc annotations
        tr = get_most_precise_tr((t1, t2), tau, temporal_rels)
        lc = get_most_precise_lc((t1, t2), tau, log_cards)
        ec = get_most_precise_ec((t1, t2), tau, event_cards)

        tr_i = get_most_precise_tr((t2, t1), tau, temporal_rels)
        lc_i = get_most_precise_lc((t2, t1), tau, log_cards)
        ec_i = get_most_precise_ec((t2, t1), tau, event_cards)

        edges[(t1, t2)] = {'TR': tr, 'LC': lc, 'EC': ec}
        edges[(t2, t1)] = {'TR': tr_i, 'LC': lc_i, 'EC': ec_i}

        if verbose:
            print(f"{t1} -> {t2}:\t TR: {tr}\t LC: {lc}\t EC: {ec}")
            print(f"{t2} -> {t1}:\t TR: {tr_i}\t LC: {lc_i}\t EC: {ec_i}")

    return nodes, edges