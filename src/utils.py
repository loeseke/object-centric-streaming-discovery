"""\
Classes and functionality for converting OCEL 2.0 log into object-centric event stream and for utility.
__author__: "Nina Löseke"
"""

import json
from typing import Any, Tuple, Union
import pandas as pd
import numpy as np
import os
import time


def min_max_normalize_dict(key_to_val : dict[Any, Union[float, int]]) -> dict[Any, float]:
    """
    Apply min-max normalization to values of given dictionary.

    Parameters
    ----------
    key_to_val : dict[Any, Union[float, int]]
        Dictionary whose values are min-max-normalized.
    
    Returns
    -------
    dict[Any, float]
        Dictionary with min-max-normalized values.
    """
    if len(key_to_val) > 0:
        vals = key_to_val.values()
        min_val = min(vals)
        max_val = max(vals)
        if max_val - min_val > 0:
            key_to_val = {key: (val - min_val)/(max_val - min_val) for key, val in key_to_val.items()}
        else:
            key_to_val = {key: 0 for key in key_to_val}
    else:
        key_to_val = dict()
    return key_to_val


def get_mean_timedelta(td_list : list[pd.Timedelta]) -> pd.Timedelta:
    """
    Computes the average time span for a list of pandas Timedeltas.

    Parameters
    ----------
    td_list : list[pd.Timedelta]
        List of time spans which is averaged component-wise to avoid overflow that may occur with built-in methods.

    Returns
    -------
    pd.Timedelta
        Average time span.
    """
    # Split computation of mean of list of Timedelta by mean calculation per unit to avoid overflow error when converting too many days via td.total_seconds()
    tds_round_to_s = [td.as_unit('s') for td in td_list]
    mean_days = round(np.mean([td.components.days for td in tds_round_to_s]))
    mean_hours = round(np.mean([td.components.hours for td in tds_round_to_s]))
    mean_minutes = round(np.mean([td.components.minutes for td in tds_round_to_s]))
    mean_seconds = round(np.mean([td.components.seconds for td in tds_round_to_s]))
    return pd.Timedelta(days=mean_days, hours=mean_hours, minutes=mean_minutes, seconds=mean_seconds)


def td_to_str(td : pd.Timedelta) -> str:
    """
    Returns a compact string corresponding to the components of the given pd.Timedelta.

    Parameters
    ----------
    td : pd.Timedelta
        Time span to represent as string.
    
    Returns
    -------
    str
        pd.Timedelta in format [days-]hours:minutes.
    """
    # Output duration as format dd-hh:mm
    td_str = f'{td.components.hours:02d}:{td.components.minutes:02d}'
    if td.components.days > 0:
        td_str = f'{td.components.days:02d}-' + td_str
    
    return td_str


class ObjectAttributeUpdate(object):
    """
    Represents an object-attribute update, which is a potential stream item in an object-centric event stream.

    Attributes
    ----------
    time : pd.Timestamp
        Time at which object-attribute update occurs.
    id : str
        Unique ID of object whose attribute is updated.
    type : str
        Type of object whose attribute is updated.
    attr : str
        Name of object attribute that is updated.
    value : Any
        New value that the object attribute receives.
    """

    def __init__(self, time : pd.Timestamp, id : str, type : str, attr : str, value : Any):
        """
        Initializes an ObjectAttributeUpdate object.

        Parameters
        ----------
        time : pd.Timestamp
            Time at which object-attribute update occurs.
        id : str
            Unique ID of object whose attribute is updated.
        type : str
            Type of object whose attribute is updated.
        attr : str
            Name of object attribute that is updated.
        value : Any
            New value that the object attribute receives.
        """
        self.time = time
        self.id = id
        self.type = type
        self.attr = attr
        self.value = value


class O2OUpdate(object):
    """
    Represents an object-to-object update, which is a potential stream item in an object-centric event stream. The qualifier of the relation between a source and target object is updated.

    Attributes
    ----------
    time : pd.Timestamp
        Time at which object-to-object update occurs.
    id : str
        Unique ID of source object.
    type : str
        Type of source object.
    target_id : str
        Unique ID of target object.
    target_type : str
        Type of target object.
    qualifier : str
        Descriptor of O2O relation between source and target object.
    """
    
    def __init__(self, time : pd.Timestamp, id : str, type : str, target_id : str, target_type : str, qualifier : str):
        """
        Initializes an O2OUpdate object.

        Parameters
        ----------
        time : pd.Timestamp
            Time at which object-to-object update occurs.
        id : str
            Unique ID of source object.
        type : str
            Type of source object.
        target_id : str
            Unique ID of target object.
        target_type : str
            Type of target object.
        qualifier : str
            Descriptor of O2O relation between source and target object.
        """
        self.time = time
        self.id = id
        self.type = type
        self.target_id = target_id
        self.target_type = target_type
        self.qualifier = qualifier
    
    def __str__(self) -> str:
        return str({
            'time': self.time,
            'id': self.id,
            'type': self.type,
            'target_id': self.target_id,
            'target_type': self.target_type,
            'qualifier': self.qualifier
        })


class Event(object):
    """
    Represents an Event object, which is a potential stream item in an object-centric event stream.

    Attributes
    ----------
    time : pd.Timestamp
        Time at which event occurs.
    id : str
        Unique ID of event, currently only used since object-centric event streams are simulated from OCEL 2.0 logs where IDs are already assigned to events.
    activity : str
        Name of activity that is performed.
    attributes : list[dict]
        List of event attributes.
    e2o_relations : list[dict]
        List of involved objects.
    """
    
    def __init__(self, time : pd.Timestamp, id : str, activity : str, attributes : list[dict[str, Any]], e2o_relations : list[dict[str, Any]]):
        """
        Initializes an Event object.

        Parameters
        ----------
        time : pd.Timestamp
            Time at which event occurs.
        id : str
            Unique ID of event, currently only used since object-centric event streams are simulated from OCEL 2.0 logs where IDs are already assigned to events.
        activity : str
            Name of activity that is performed.
        attributes : list[dict[str, Any]]
            List of event attributes.
        e2o_relations : list[dict[str, Any]]
            List of involved objects.
        """
        self.time = time
        self.id = id
        self.activity = activity
        self.attributes = attributes
        self.e2o_relations = e2o_relations


def parse_type_per_oid_json(objects_dict : list[dict[str, str]]) -> dict[str, str]:
    """
    Creates mapping of objects to types for given list of objects.

    Parameters
    ----------
    objects_dict : list[dict[str, str]]

    Returns
    -------
    dict[str, str]
        Mapping of objects to corresponding object types.
    """
    type_per_obj_id = dict()
    
    for obj in objects_dict:
        type_per_obj_id[obj['id']] = obj['type']
    
    return type_per_obj_id


def parse_ocel2_json_object_updates(objects_dict : list[dict[str, str]]) -> list[ObjectAttributeUpdate]:
    """
    Creates time-ordered list of ObjectAttributeUpdates from specification of objects according to OCEL 2.0 JSON exchange format.

    Parameters
    ----------
    objects_dict : list[dict[str, str]]
        List of objects from OCEL 2.0 JSON file.
    
    Returns
    -------
    list[ObjectAttributeUpdate]
        List of derived ObjectAttributeUpdates ordered by time from least to most recent.
    """
    # Assume OCEL 2.0 JSON as input format
    object_updates = list()

    # Parse object-attribute updates
    for obj in objects_dict:
        # Object attributes field is not guaranteed, e.g. for Age of Empires OCEL
        if 'attributes' in obj:
            for obj_attr in obj['attributes']:
                object_updates.append(
                    ObjectAttributeUpdate(
                        time=pd.to_datetime(obj_attr['time']), 
                        id=obj['id'], 
                        type=obj['type'],
                        attr=obj_attr['name'],
                        value=obj_attr['value']
                    )
                )

    object_updates = sorted(object_updates, key=lambda x: x.time)
    return object_updates
    

def parse_ocel2_json_o2o_updates(objects_dict : list[dict[str, str]], min_ts_per_obj_id : dict[str, pd.Timestamp], o2o_has_time : bool = False, verbose : bool = False) -> list[O2OUpdate]:
    """
    Creates time-ordered list of O2OUpdates from specification of objects according to OCEL 2.0 JSON exchange format.
    
    Parameters
    ----------
    objects_dict : list[dict[str, str]]
        List of objects from OCEL 2.0 JSON file.
    min_ts_per_obj_id : dict[str, pd.Timestamp]
        Mapping of objects to timestamps of their first occurrences in the log.
    o2o_has_time : bool, default=False
        If True, an associated timestamp is stored along object-to-object updates in the OCEL 2.0 JSON file. If False, the timestamp is derived from the source object's initial occurrence.
    verbose : bool, default=False
        If enabled, information about parsed log is printed.

    Returns
    -------
    list[O2OUpdate]
        List of derived O2OUpdates ordered by time from least to most recent.
    """
    # Assume OCEL 2.0 JSON as input format
    o2o_updates = list()
    type_per_obj_id = parse_type_per_oid_json(objects_dict)

    # Check for objects that only occur in O2O relations and are not also specified in "objects" field of JSON along w/ their attributes and type
    obj_ids_wo_type = list()

    for obj in objects_dict:
        id = obj['id']
        type = obj['type']

        if 'relationships' in obj:
            for o2o_rel in obj['relationships']:
                target_id = o2o_rel['objectId']

                # For O2O updates, assume timestamp of attribute that is set first
                # Only add O2O relation where type can be derived
                if target_id in type_per_obj_id:
                    o2o_updates.append(
                        O2OUpdate(
                            # If no timestamp for O2O is given (it is not as per OCEL 2.0 standard), use timestamp of first event involving source object ID
                            time=min_ts_per_obj_id[id] if not o2o_has_time else pd.to_datetime(o2o_rel['time']),
                            id=id,
                            type=type,
                            target_id=target_id,
                            target_type=type_per_obj_id[target_id],
                            qualifier=o2o_rel['qualifier']
                        )
                    )
                else:
                    obj_ids_wo_type.append(target_id)
    
    if verbose:
        print(f'# of (removed) O2O target objects w/o type: {len(set(obj_ids_wo_type))}')

    o2o_updates = sorted(o2o_updates, key=lambda x: x.time)
    return o2o_updates


def parse_ocel2_json_events(events_dict : list[dict[str, Any]], objects_dict : list[dict[str, Any]], verbose : bool = False) -> Tuple[list[Event], list[O2OUpdate], dict[str, pd.Timestamp]]:
    """
    Creates time-ordered list of Events from specification of objects according to OCEL 2.0 JSON exchange format. Additionally, the timestamp of the first occurrence of an object in an event is extracted. "Enriched" object-to-object updates derived from the event-to-object relations are also defined.

    Parameters
    ----------
    events_dict : list[dict[str, Any]]
        List of events from OCEL 2.0 JSON file.
    objects_dict : list[dict[str, Any]]
        List of objects from OCEL 2.0 JSON file.
    verbose : bool, default=False
        If enabled, information about parsed log is printed.

    Returns
    -------
    Tuple[list[Event], list[O2OUpdate], dict[str, pd.Timestamp]]
        Time-ordered list of events, object-to-object updates derived from events, and mapping of objects to timestamps of first occurrences in events.
    """
    events = list()
    enriched_o2o_updates = list()

    min_ts_per_obj_id = dict()
    type_per_obj_id = parse_type_per_oid_json(objects_dict)
    e2o_oids_wo_type = list()

    for ev in events_dict:
        time = pd.to_datetime(ev['time'])

        # Track minimal timestamp of event where each object first occurs
        e2o_rel_cleaned = list()
        if 'relationships' in ev:
            for e2o_rel in ev['relationships']:
                obj_id = e2o_rel['objectId']

                if obj_id not in type_per_obj_id:
                    e2o_oids_wo_type.append(obj_id)
                else:
                    # JSON only stores object ID per event instead of type; add available object type
                    e2o_rel['objectType'] = type_per_obj_id[obj_id]
                    e2o_rel_cleaned.append(e2o_rel)

                if obj_id in min_ts_per_obj_id:
                    min_ts_per_obj_id[obj_id] = min(time, min_ts_per_obj_id[obj_id])
                else: 
                    min_ts_per_obj_id[obj_id] = time

        # Parse event updates; filter E2O relations to ensure each object involved has type
        events.append(
            Event(
                time=time,
                id=ev['id'],
                activity=ev['type'],
                # Event attributes field isn't guaranteed, e.g. for Age of Empires OCEL
                attributes=ev['attributes'] if 'attributes' in ev else list(),
                e2o_relations=e2o_rel_cleaned
            )
        )

        # Add enriched O2O updates derived from object pairs participating in same event
        for e2o_rel_source, e2o_rel_target in list(zip(e2o_rel_cleaned, e2o_rel_cleaned[1:] + e2o_rel_cleaned[:1])):
            enriched_o2o_updates.append(
                O2OUpdate(
                    time=time,
                    id=e2o_rel_source['objectId'],
                    type=e2o_rel_source['objectType'],
                    target_id=e2o_rel_target['objectId'],
                    target_type=e2o_rel_target['objectType'],
                    # Set event activity name as qualifier of derived O2O relation
                    qualifier=ev['type']
                )
            )

    if verbose:
        print(f'# of (removed) E2O target objects w/o type: {len(set(e2o_oids_wo_type))}')

    events.sort(key=lambda x: x.time)
    enriched_o2o_updates.sort(key=lambda x: x.time)
    return events, enriched_o2o_updates, min_ts_per_obj_id


def parse_ocel2_json(file_path : str, o2o_has_time : bool = False, verbose : bool = False) -> Tuple[list[Event], list[ObjectAttributeUpdate], list[O2OUpdate], list[O2OUpdate], list[str]]:
    """
    Extracts time-ordered lists of events, (enriched) object-to-object-updates, and object-attribute updates from given OCEL 2.0 in JSON exchange format. Additionally, the object types are extracted.

    Parameters
    ----------
    file_path : str
        Path to OCEL 2.0 JSON file.
    o2o_has_time : bool, default=False
        If True, an associated timestamp is stored along object-to-object updates in the OCEL 2.0 JSON file. If False, the timestamp is derived from the source object's initial occurrence.
    verbose : bool, default=False
        If enabled, information about parsed log is printed.

    Returns
    -------
    Tuple[list[Event], list[ObjectAttributeUpdate], list[O2OUpdate], list[O2OUpdate], list[str]]
        Time-ordered lists of events, object-attribute updates, object-to-object updates, "enriched" object-to-object updates, and list of object types.
    """
    with open(file_path, 'r') as ocel_file:
        # NOTE: OCEL JSON dict has keys 'objectTypes', 'eventTypes', 'objects', 'events'
        ocel_dict = json.load(ocel_file)

    events, enriched_o2o_updates, min_ts_per_obj_id = parse_ocel2_json_events(ocel_dict['events'], ocel_dict['objects'], verbose)
    object_updates = parse_ocel2_json_object_updates(ocel_dict['objects'])
    o2o_updates = parse_ocel2_json_o2o_updates(ocel_dict['objects'], min_ts_per_obj_id, o2o_has_time, verbose)

    return events, object_updates, o2o_updates, enriched_o2o_updates, [ot_dict['name'] for ot_dict in ocel_dict['objectTypes']]


class EventStream(object):
    """
    Represents an object-centric event stream based on the OCEL 2.0 metamodel for a given OCEL 2.0 log.

    Attributes
    ----------
    events : list[Event]
        Time-ordered list of events.
    object_updates : list[ObjectAttributeUpdate]
        Time-ordered list of object-attribute updates.
    o2o_updates : list[O2OUpdate]
        Time-ordered list of object-to-object updates.
    enriched_o2o_updates : list[O2OUpdate]
        Time-ordered list of object-to-object updates derived from event-to-object relations.
    object_types : list[str]
        Unique object types occurring in log.
    stream : list[Union[Event, O2OUpdate, ObjectAttributeUpdate]]
        Simulates object-centric event stream as time-ordered lists of events, object-attribute updates, or object-to-object updates as stream items.
    """
    
    def __init__(self, file_path : str, enrich_o2o : bool = False, o2o_has_time : bool = False, verbose : bool = False):
        """
        Creates an object-centric event stream for a given OCEL 2.0 log in JSON exchange format.

        Parameters
        ----------
        file_path : str
            Path of OCEL 2.0 JSON file.
        enrich_o2o : bool, default=False
            If True, object-to-object updates derived from event-to-object relations is added into stream. Alternatively, these "enriched" O2OUpdates can be added to the stream during stream processing for incoming events.
        o2o_has_time : bool, default=False
            If True, an associated timestamp is stored along object-to-object updates in the OCEL 2.0 JSON file. If False, the timestamp is derived from the source object's initial occurrence.
        verbose : bool, default=False
            If enabled, information about parsed log is printed.
        """
        if verbose:
            print(f'Parsing {file_path}...')
            start_time = time.time()

        self.events, self.object_updates, self.o2o_updates, self.enriched_o2o_updates, self.object_types = parse_ocel2_json(file_path, o2o_has_time, verbose)

        # Print statistics for parsed event log
        if verbose:
            print(f'# events:\t\t\t{len(self.events)}\n# object updates:\t\t{len(self.object_updates)}\n# O2O relations:\t\t{len(self.o2o_updates)}\n# E2O-derived O2O relations:\t{len(self.enriched_o2o_updates)}\nEnriching enabled: {enrich_o2o}')

        # Represent "stream" as list of Event, ObjectAttributeUpdate, O2OUpdate, and E2OUpdate objects sorted by timestamp
        if not enrich_o2o:
            self.stream = sorted(self.events + self.object_updates + self.o2o_updates, key=lambda x: x.time)
        else:
            self.stream = sorted(self.events + self.object_updates + self.o2o_updates + self.enriched_o2o_updates, key=lambda x: x.time)
        
        if verbose:
            print(f'Finished parsing {file_path} in {(time.time()-start_time)/60:.2f} min.')
    
    def create_stream_chunks(self) -> dict[int, list[Union[Event, O2OUpdate, ObjectAttributeUpdate]]]:
        """
        Splits object-centric event stream into 10% chunks s.t. each chunk contains 10% of stream items.

        Returns
        -------
        dict[int, list[Union[Event, O2OUpdate, ObjectAttributeUpdate]]]
            Mapping of percentages to chunks of stream items.
        """
        pct_range = range(0, 100, 10)
        pct_to_chunk = {pct: list() for pct in pct_range[1:]}
        stream_size = len(self.stream)
        eval_stream_percentages = [(pct, int(stream_size*pct/100)) for pct in pct_range]
        
        for i in range(1, len(eval_stream_percentages)):
            _, stream_item_lb = eval_stream_percentages[i-1]
            pct, stream_item_ub = eval_stream_percentages[i]
            pct_to_chunk[pct] = self.stream[stream_item_lb:stream_item_ub]
        pct_to_chunk[100] = self.stream[eval_stream_percentages[-1][1]:]
        
        return pct_to_chunk