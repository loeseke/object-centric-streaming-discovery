"""\
Functionality for extracting object-centric characteristics used by priority policies from full offline log.
__author__: "Nina Löseke"
"""

from utils import *
import numpy as np


def define_skewed_stream() -> list:
    td_start = pd.Timestamp(year=2023, month=1, day=1)
    td_end = pd.Timestamp(year=2024, month=1, day=1)
    td_curr = td_start
    td_ot_1 = pd.Timedelta(days=30) # pd.Timedelta has no "months" keyword
    td_ot_2 = pd.Timedelta(weeks=1)
    td_ot_3 = pd.Timedelta(days=1)
    events = list()
    event_id = 0

    # Have weekly & monthly OTs be updated on single object each throughout entire year
    while td_curr < td_end:
        new_event = Event(td_curr, event_id, activity='Register object', attributes=[], e2o_relations=[{'objectType': 'Monthly', 'objectId': 'm1'}])
        events.append(new_event)

        td_curr += td_ot_1
        event_id += 1

    td_curr = td_start
    while td_curr < td_end:
        new_event = Event(td_curr, event_id, activity='Register object', attributes=[], e2o_relations=[{'objectType': 'Weekly', 'objectId': 'w1'}])
        events.append(new_event)

        td_curr += td_ot_2
        event_id += 1

    # Only have object type 3 occur between May and August (2nd third of the year)
    td_start_ot_3 = pd.Timestamp(year=2023, month=5, day=1)
    td_end_ot_3 = pd.Timestamp(year=2023, month=9, day=1)
    td_curr_ot_3 = td_start_ot_3
    step = 1
    ot_3_obj_id = 1
    while td_curr_ot_3 < td_end_ot_3:
        new_event = Event(td_curr_ot_3, event_id, activity='Register object', attributes=[], e2o_relations=[{'objectType': 'Daily', 'objectId': f'd{ot_3_obj_id}'}])
        events.append(new_event)

        if step % 2 == 0:
            ot_3_obj_id += 1
            step = 1
        else:
            step += 1
        td_curr_ot_3 += td_ot_3
        event_id += 1
    
    stream = sorted(events, key=lambda x: x.time)
    return stream
            

def __fraction_to_interval_str(frac : float, stride : float = 0.1) -> str:
    upper_bound = str(round(frac * 100, 1))
    lower_bound = str(round((frac-stride) * 100, 1))
    return f'({lower_bound}, {upper_bound}]'


def get_ot_frac_across_stream(stream : list) -> Tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    ot_to_freq_total = dict()
    ot_to_freq_event = dict()
    ot_to_freq_o2o = dict()
    ot_to_freq_oau = dict()

    freq_total = 0
    freq_event = 0
    freq_o2o = 0
    freq_oau = 0

    for stream_item in stream:
        freq_total += 1

        if isinstance(stream_item, Event):
            ots = set([d['objectType'] for d in stream_item.e2o_relations])
            freq_event += 1

            for ot in ots:
                ot_to_freq_event.setdefault(ot, 0)
                ot_to_freq_event[ot] += 1
        elif isinstance(stream_item, O2OUpdate):
            ots = set([stream_item.type, stream_item.target_type])
            freq_o2o += 1

            for ot in ots:
                ot_to_freq_o2o.setdefault(ot, 0)
                ot_to_freq_o2o[ot] += 1
        elif isinstance(stream_item, ObjectAttributeUpdate):
            ots = [stream_item.type]
            freq_oau += 1

            for ot in ots:
                ot_to_freq_oau.setdefault(ot, 0)
                ot_to_freq_oau[ot] += 1
        else:
            raise RuntimeError('Unknown type of EventStream item encountered!')
        
        for ot in ots:
            ot_to_freq_total.setdefault(ot, 0)
            ot_to_freq_total[ot] += 1
        
    ot_to_freq_total = {ot: freq / freq_total for ot, freq in ot_to_freq_total.items()}
    ot_to_freq_event = {ot: freq / freq_total for ot, freq in ot_to_freq_event.items()}
    ot_to_freq_o2o = {ot: freq / freq_total for ot, freq in ot_to_freq_o2o.items()}
    ot_to_freq_oau = {ot: freq / freq_total for ot, freq in ot_to_freq_oau.items()}

    return ot_to_freq_total, ot_to_freq_event, ot_to_freq_o2o, ot_to_freq_oau


def get_num_obj_per_event_per_ot(stream : list) -> Tuple[dict, dict, dict, dict]:
    n_stream = len(stream)
    t_min_stream = stream[0].time
    t_max_stream = stream[-1].time
    td_stream = t_max_stream - t_min_stream

    ot_dict = dict()
    frac_keys = [__fraction_to_interval_str(frac) for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
    ot_dict_per_stream_fraction = {frac: dict() for frac in frac_keys}
    ot_dict_per_time_fraction = {frac: dict() for frac in frac_keys}
    res_dict = dict()
    res_dict_per_stream_fraction = {frac: dict() for frac in frac_keys}
    res_dict_per_time_fraction = {frac: dict() for frac in frac_keys}

    for i, item in enumerate(stream):
        if isinstance(item, Event):
            frac_item = np.ceil((i+1)/n_stream * 10)/10
            frac_item_str = __fraction_to_interval_str(frac_item, 0.1)

            td_since_start = item.time - t_min_stream
            frac_td = np.ceil(td_since_start/td_stream * 10)/10 if td_since_start.total_seconds() > 0 else 0.1
            frac_td_str = __fraction_to_interval_str(frac_td)

            event_ot_freq = dict()
            for e2o_dict in item.e2o_relations:
                ot = e2o_dict['objectType']
                if ot not in event_ot_freq:
                    event_ot_freq[ot] = 0
                event_ot_freq[ot] += 1

            for ot in event_ot_freq:
                if ot not in ot_dict:
                    ot_dict[ot] = list()
                ot_dict[ot].append(event_ot_freq[ot])

                if ot not in ot_dict_per_stream_fraction[frac_item_str]:
                    ot_dict_per_stream_fraction[frac_item_str][ot] = list()
                ot_dict_per_stream_fraction[frac_item_str][ot].append(event_ot_freq[ot])

                if ot not in ot_dict_per_time_fraction[frac_td_str]:
                    ot_dict_per_time_fraction[frac_td_str][ot] = list()
                ot_dict_per_time_fraction[frac_td_str][ot].append(event_ot_freq[ot])
        else:
            continue
    
    # Average counts of OT occurences per event across entire stream, item buckets, and time buckets
    for ot in ot_dict:
        res_dict[ot] = np.mean(ot_dict[ot]).item() if ot in ot_dict else 0
        
        for frac in frac_keys:
            res_dict_per_stream_fraction[frac][ot] = np.mean(ot_dict_per_stream_fraction[frac][ot]).item() if ot in ot_dict_per_stream_fraction[frac] else 0
            res_dict_per_time_fraction[frac][ot] = np.mean(ot_dict_per_time_fraction[frac][ot]).item() if ot in ot_dict_per_time_fraction[frac] else 0
    
    return ot_dict, res_dict, res_dict_per_stream_fraction, res_dict_per_time_fraction


def get_num_obj_per_ot(stream : list) -> Tuple[dict, dict, dict]:
    n_stream = len(stream)
    t_min_stream = stream[0].time
    t_max_stream = stream[-1].time
    td_stream = t_max_stream - t_min_stream

    ot_dict = dict()
    frac_keys = [__fraction_to_interval_str(frac) for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
    ot_dict_per_stream_fraction = {frac: dict() for frac in frac_keys}
    ot_dict_per_time_fraction = {frac: dict() for frac in frac_keys}
    res_dict = dict()
    res_dict_per_stream_fraction = {frac: dict() for frac in frac_keys}
    res_dict_per_time_fraction = {frac: dict() for frac in frac_keys}

    for i, item in enumerate(stream):
        if isinstance(item, Event) or isinstance(item, O2OUpdate):
            frac_item = np.ceil((i+1)/n_stream * 10)/10
            frac_item_str = __fraction_to_interval_str(frac_item, 0.1)
            
            td_since_start = item.time - t_min_stream
            frac_td = np.ceil(td_since_start/td_stream * 10)/10 if td_since_start.total_seconds() > 0 else 0.1
            frac_td_str = __fraction_to_interval_str(frac_td)

            ot_obj_pairs = list()
            if isinstance(item, Event):
                for e2o_dict in item.e2o_relations:
                    ot = e2o_dict['objectType']
                    obj = e2o_dict['objectId']

                    ot_obj_pairs.append((ot, obj))
            else:
                ot_obj_pairs.append((item.type, item.id))
                ot_obj_pairs.append((item.target_type, item.target_id))

            for ot, obj in ot_obj_pairs:
                if ot not in ot_dict:
                    ot_dict[ot] = set()
                ot_dict[ot].add(obj)

                if ot not in ot_dict_per_stream_fraction[frac_item_str]:
                    ot_dict_per_stream_fraction[frac_item_str][ot] = set()
                ot_dict_per_stream_fraction[frac_item_str][ot].add(obj)

                if ot not in ot_dict_per_time_fraction[frac_td_str]:
                    ot_dict_per_time_fraction[frac_td_str][ot] = set()
                ot_dict_per_time_fraction[frac_td_str][ot].add(obj)
        else:
            continue
    
    # Count unique object occurrences across entire stream, item buckets, and time buckets
    for ot in ot_dict:
        res_dict[ot] = len(ot_dict[ot]) if ot in ot_dict else 0
        
        for frac in frac_keys:
            res_dict_per_stream_fraction[frac][ot] = len(ot_dict_per_stream_fraction[frac][ot]) if ot in ot_dict_per_stream_fraction[frac] else 0
            res_dict_per_time_fraction[frac][ot] = len(ot_dict_per_time_fraction[frac][ot]) if ot in ot_dict_per_time_fraction[frac] else 0
    
    return res_dict, res_dict_per_stream_fraction, res_dict_per_time_fraction


def get_num_events_per_ot(stream : list) -> Tuple[dict, dict, dict]:
    n_stream = len(stream)
    t_min_stream = stream[0].time
    t_max_stream = stream[-1].time
    td_stream = t_max_stream - t_min_stream

    ot_dict = dict()
    frac_keys = [__fraction_to_interval_str(frac) for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
    ot_dict_per_stream_fraction = {frac: dict() for frac in frac_keys}
    ot_dict_per_time_fraction = {frac: dict() for frac in frac_keys}

    for i, item in enumerate(stream):
        if isinstance(item, Event):
            frac_item = np.ceil((i+1)/n_stream * 10)/10
            frac_item_str = __fraction_to_interval_str(frac_item, 0.1)
            
            td_since_start = item.time - t_min_stream
            frac_td = np.ceil(td_since_start/td_stream * 10)/10 if td_since_start.total_seconds() > 0 else 0.1
            frac_td_str = __fraction_to_interval_str(frac_td)

            ot_set = set()
            if isinstance(item, Event):
                for e2o_dict in item.e2o_relations:
                    ot_set.add(e2o_dict['objectType'])
            else:
                ot_set.add(item.type)
                ot_set.add(item.target_type)

            for ot in ot_set:
                if ot not in ot_dict:
                    ot_dict[ot] = 0
                ot_dict[ot] += 1

                if ot not in ot_dict_per_stream_fraction[frac_item_str]:
                    ot_dict_per_stream_fraction[frac_item_str][ot] = 0
                ot_dict_per_stream_fraction[frac_item_str][ot] += 1

                if ot not in ot_dict_per_time_fraction[frac_td_str]:
                    ot_dict_per_time_fraction[frac_td_str][ot] = 0
                ot_dict_per_time_fraction[frac_td_str][ot] += 1
        else:
            continue
    
    # Fill in zeroes if certain OT does not occur at all in a stream-item or time bucket
    for ot in ot_dict.keys():
        for frac in frac_keys:
            if ot not in ot_dict_per_stream_fraction[frac]:
                ot_dict_per_stream_fraction[frac][ot] = 0
            if ot not in ot_dict_per_time_fraction[frac]:
                ot_dict_per_time_fraction[frac][ot] = 0
    
    return ot_dict, ot_dict_per_stream_fraction, ot_dict_per_time_fraction


def __get_mean_timedelta(td_list : list) -> pd.Timedelta:
    # Split computation of mean of list of Timedelta by mean calculation per unit to avoid overflow error when converting too many days via td.total_seconds()
    tds_round_to_s = [td.as_unit('s') for td in td_list]
    mean_days = round(np.mean([td.components.days for td in tds_round_to_s]))
    mean_hours = round(np.mean([td.components.hours for td in tds_round_to_s]))
    mean_minutes = round(np.mean([td.components.minutes for td in tds_round_to_s]))
    mean_seconds = round(np.mean([td.components.seconds for td in tds_round_to_s]))
    return pd.Timedelta(days=mean_days, hours=mean_hours, minutes=mean_minutes, seconds=mean_seconds)


def seconds_to_td_str(td_total_seconds : float) -> str:
    td = pd.Timedelta(seconds=td_total_seconds)
    ret = f'{td.components.hours:02d}:{td.components.minutes:02d}'
    if td.components.days > 0:
        ret = f'{td.components.days:02d}-' + ret
    return ret


def get_reuse_stride(stream : list) -> Tuple[dict, dict, dict, dict]:
    n_stream = len(stream)
    t_min_stream = stream[0].time
    t_max_stream = stream[-1].time
    td_stream = t_max_stream - t_min_stream

    last_seen_dict = dict()
    ot_dict = dict()
    frac_keys = [__fraction_to_interval_str(frac) for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
    ot_dict_per_stream_fraction = {frac: dict() for frac in frac_keys}
    ot_dict_per_time_fraction = {frac: dict() for frac in frac_keys}
    avg_per_obj = dict()
    avg_per_ot = dict()
    ot_avg_per_stream_fraction = {frac: dict() for frac in frac_keys}
    ot_avg_per_time_fraction = {frac: dict() for frac in frac_keys}

    for i, item in enumerate(stream):
        if isinstance(item, Event):
            frac_item = np.ceil((i+1)/n_stream * 10)/10
            frac_item_str = __fraction_to_interval_str(frac_item, 0.1)
            
            td_since_start = item.time - t_min_stream
            frac_td = np.ceil(td_since_start/td_stream * 10)/10 if td_since_start.total_seconds() > 0 else 0.1
            frac_td_str = __fraction_to_interval_str(frac_td)

            ot_obj_pairs = list()
            if isinstance(item, Event):
                for e2o_dict in item.e2o_relations:
                    ot_obj_pairs.append((e2o_dict['objectType'], e2o_dict['objectId']))
            else:
                ot_obj_pairs.append((item.type, item.id))
                ot_obj_pairs.append((item.target_type, item.target_id))

            for ot, obj in ot_obj_pairs:
                if ot not in ot_dict:
                    ot_dict[ot] = dict()
                if obj not in ot_dict[ot]:
                    ot_dict[ot][obj] = list()
                if ot not in ot_dict_per_stream_fraction[frac_item_str]:
                    ot_dict_per_stream_fraction[frac_item_str][ot] = dict()
                if obj not in ot_dict_per_stream_fraction[frac_item_str][ot]:
                    ot_dict_per_stream_fraction[frac_item_str][ot][obj] = list()
                if ot not in ot_dict_per_time_fraction[frac_td_str]:
                    ot_dict_per_time_fraction[frac_td_str][ot] = dict()
                if obj not in ot_dict_per_time_fraction[frac_td_str][ot]:
                    ot_dict_per_time_fraction[frac_td_str][ot][obj] = list()
                if ot not in last_seen_dict:
                    last_seen_dict[ot] = dict()

                if obj not in last_seen_dict[ot]:
                    last_seen_dict[ot][obj] = item.time
                else:
                    td_item = item.time - last_seen_dict[ot][obj]
                    ot_dict[ot][obj].append(td_item)
                    ot_dict_per_stream_fraction[frac_item_str][ot][obj].append(td_item)
                    ot_dict_per_time_fraction[frac_td_str][ot][obj].append(td_item)
                    last_seen_dict[ot][obj] = item.time
        else:
            continue

    # Compute average stride of re-occurrence per object per OT vs. per OT overall
    for ot in ot_dict:
        ot_all_obj_td = list()
        for obj in ot_dict[ot]:
            if ot not in avg_per_obj:
                avg_per_obj[ot] = list()
            
            if len(ot_dict[ot][obj]) < 1:
                continue
            else:
                avg_per_obj[ot].append(__get_mean_timedelta(ot_dict[ot][obj]).total_seconds())
                ot_all_obj_td += ot_dict[ot][obj]
        if len(ot_all_obj_td) > 0:
            avg_per_ot[ot] = __get_mean_timedelta(ot_all_obj_td).total_seconds()
        
        for frac in frac_keys:
            if ot in ot_dict_per_stream_fraction[frac]: 
                frac_ot_all_obj_td = list()
                for obj in ot_dict_per_stream_fraction[frac][ot]:
                    frac_ot_all_obj_td += ot_dict_per_stream_fraction[frac][ot][obj]
                if len(frac_ot_all_obj_td) > 0:
                    ot_avg_per_stream_fraction[frac][ot] = __get_mean_timedelta(frac_ot_all_obj_td).total_seconds()

            if ot in ot_dict_per_time_fraction[frac]:
                frac_ot_all_obj_td = list()
                for obj in ot_dict_per_time_fraction[frac][ot]:
                    frac_ot_all_obj_td += ot_dict_per_time_fraction[frac][ot][obj]
                if len(frac_ot_all_obj_td) > 0:
                    ot_avg_per_time_fraction[frac][ot] = __get_mean_timedelta(frac_ot_all_obj_td).total_seconds()
    
    return avg_per_obj, avg_per_ot, ot_avg_per_stream_fraction, ot_avg_per_time_fraction


def get_lifespan(stream : list) -> Tuple[dict, dict]:
    ot_dict = dict()
    ls_per_obj = dict()
    avg_per_ot = dict()

    for item in stream:
        if isinstance(item, Event): # or isinstance(item, O2OUpdate):
            ot_obj_pairs = list()
            if isinstance(item, Event):
                for e2o_dict in item.e2o_relations:
                    ot_obj_pairs.append((e2o_dict['objectType'], e2o_dict['objectId']))
            else:
                ot_obj_pairs.append((item.type, item.id))
                ot_obj_pairs.append((item.target_type, item.target_id))

            for ot, obj in ot_obj_pairs:
                if ot not in ot_dict:
                    ot_dict[ot] = dict()
                if obj not in ot_dict[ot]:
                    ot_dict[ot][obj] = {'first': item.time, 'last': item.time}
                else:
                    ot_dict[ot][obj]['last'] = item.time
        else:
            continue

    # Average lifespans per object per OT vs. per OT generally
    for ot in ot_dict:
        if ot not in ls_per_obj:
            ls_per_obj[ot] = list()
        for obj in ot_dict[ot]:
            lifespan_obj = ot_dict[ot][obj]['last'] - ot_dict[ot][obj]['first']
            ls_per_obj[ot].append(lifespan_obj)
        
        avg_per_ot[ot] = __get_mean_timedelta(ls_per_obj[ot]).total_seconds()
    
        # Convert per-object time deltas to total seconds
        ls_per_obj[ot] = [td_obj.total_seconds() for td_obj in ls_per_obj[ot]]

    return ls_per_obj, avg_per_ot


def get_co_occurrence(stream : list) -> Tuple[dict, dict, dict]:
    ot_dict = dict()

    for item in stream:
        if isinstance(item, Event):
            involved_ots = set()
            for e2o_dict in item.e2o_relations:
                involved_ots.add(e2o_dict['objectType'])

            for ot in involved_ots:
                for ot_target in involved_ots:
                    if ot == ot_target:
                        continue

                    if ot not in ot_dict:
                        ot_dict[ot] = dict()
                    if ot_target not in ot_dict[ot]:
                        ot_dict[ot][ot_target] = 0
                    
                    ot_dict[ot][ot_target] += 1
        else:
            continue
    
    # Fill in zeroes
    for ot in ot_dict.keys():
        for ot_target in ot_dict.keys():
            if ot == ot_target:
                continue

            if ot_target not in ot_dict[ot]:
                ot_dict[ot][ot_target] = 0
    
    return ot_dict