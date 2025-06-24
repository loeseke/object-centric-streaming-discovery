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


def get_mean_timedelta(td_list : list) -> pd.Timedelta:
    # Split computation of mean of list of Timedelta by mean calculation per unit to avoid overflow error when converting too many days via td.total_seconds()
    tds_round_to_s = [td.as_unit('s') for td in td_list]
    mean_days = round(np.mean([td.components.days for td in tds_round_to_s]))
    mean_hours = round(np.mean([td.components.hours for td in tds_round_to_s]))
    mean_minutes = round(np.mean([td.components.minutes for td in tds_round_to_s]))
    mean_seconds = round(np.mean([td.components.seconds for td in tds_round_to_s]))
    return pd.Timedelta(days=mean_days, hours=mean_hours, minutes=mean_minutes, seconds=mean_seconds)


def td_to_str(td : pd.Timedelta) -> str:
    # Output duration as format dd-hh:mm
    td_str = f'{td.components.hours:02d}:{td.components.minutes:02d}'
    if td.components.days > 0:
        td_str = f'{td.components.days:02d}-' + td_str
    
    return td_str


class ObjectAttributeUpdate(object):
    def __init__(self, time : pd.Timestamp, id : str, type : str, attr : str, value : Any):
        self.time = time
        self.id = id
        self.type = type
        self.attr = attr
        self.value = value


class O2OUpdate(object):
    def __init__(self, time : pd.Timestamp, id : str, type : str, target_id : str, target_type : str, qualifier : str):
        self.time = time
        self.id = id
        self.type = type
        self.target_id = target_id
        self.target_type = target_type
        self.qualifier = qualifier
    
    def __str__(self):
        return str({
            'time': self.time,
            'id': self.id,
            'type': self.type,
            'target_id': self.target_id,
            'target_type': self.target_type,
            'qualifier': self.qualifier
        })


class Event(object):
    def __init__(self, time : pd.Timestamp, id : str, activity : str, attributes : list[dict], e2o_relations : list[dict]):
        self.time = time
        self.id = id
        self.activity = activity
        self.attributes = attributes
        self.e2o_relations = e2o_relations


def parse_type_per_oid_json(objects_dict : dict) -> dict:
    type_per_obj_id = dict()
    
    for obj in objects_dict:
        type_per_obj_id[obj['id']] = obj['type']
    
    return type_per_obj_id


def parse_ocel2_json_object_updates(objects_dict : dict) -> Tuple[dict, dict]:
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
    

def parse_ocel2_json_o2o_updates(objects_dict : dict, min_ts_per_obj_id : dict, o2o_has_time : bool = False) -> Tuple[dict, dict]:
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
    
    print(f'# of (removed) O2O target objects w/o type: {len(set(obj_ids_wo_type))}')

    o2o_updates = sorted(o2o_updates, key=lambda x: x.time)
    return o2o_updates


def parse_ocel2_json_events(events_dict : dict, objects_dict : dict) -> Tuple[list, dict]:
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

    print(f'# of (removed) E2O target objects w/o type: {len(set(e2o_oids_wo_type))}')

    events.sort(key=lambda x: x.time)
    enriched_o2o_updates.sort(key=lambda x: x.time)
    return events, enriched_o2o_updates, min_ts_per_obj_id


def parse_ocel2_json(file_path : str, o2o_has_time : bool = False) -> Tuple[list, list, list, list, list]:
    with open(file_path, 'r') as ocel_file:
        # NOTE: OCEL JSON dict has keys 'objectTypes', 'eventTypes', 'objects', 'events'
        ocel_dict = json.load(ocel_file)

    events, enriched_o2o_updates, min_ts_per_obj_id = parse_ocel2_json_events(ocel_dict['events'], ocel_dict['objects'])
    object_updates = parse_ocel2_json_object_updates(ocel_dict['objects'])
    o2o_updates = parse_ocel2_json_o2o_updates(ocel_dict['objects'], min_ts_per_obj_id, o2o_has_time)

    return events, object_updates, o2o_updates, enriched_o2o_updates, [ot_dict['name'] for ot_dict in ocel_dict['objectTypes']]


class EventStream(object):
    def __init__(self, file_path : str, enrich_o2o : bool = False, o2o_has_time : bool = False):
        print(f'Parsing {file_path}...')
        start_time = time.time()

        self.events, self.object_updates, self.o2o_updates, self.enriched_o2o_updates, self.object_types = parse_ocel2_json(file_path, o2o_has_time)

        # Print statistics for parsed event log
        print(f'# events:\t\t\t{len(self.events)}\n# object updates:\t\t{len(self.object_updates)}\n# O2O relations:\t\t{len(self.o2o_updates)}\n# E2O-derived O2O relations:\t{len(self.enriched_o2o_updates)}\nEnriching enabled: {enrich_o2o}')

        # Represent "stream" as list of Event, ObjectAttributeUpdate, O2OUpdate, and E2OUpdate objects sorted by timestamp
        if not enrich_o2o:
            self.stream = sorted(self.events + self.object_updates + self.o2o_updates, key=lambda x: x.time)
        else:
            self.stream = sorted(self.events + self.object_updates + self.o2o_updates + self.enriched_o2o_updates, key=lambda x: x.time)
        
        print(f'Finished parsing {file_path} in {(time.time()-start_time)/60:.2f} min.')
    
    def create_stream_chunks(self) -> dict[int, list[Event | O2OUpdate | ObjectAttributeUpdate]]:
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


# Test log-to-stream parsing with OCEL 2.0 JSON
if __name__ == "__main__":
    es = EventStream(os.path.join("../data", "ContainerLogistics.json"), o2o_has_time=False)
    es.create_stream_chunks()