"""\
Classes for streaming representations of OC-DFGs, OCPNs, and TOTeM models based on model buffers, cache polices, and optional priority policies.
__author__: "Nina Löseke"
"""

import os
import itertools
from vars import *
from typing import Any, Tuple, Union
from copy import deepcopy
import pandas as pd
import time as time
from monitor import CacheMonitor, RuntimeMonitor
from cache_policy_buffers import CachePolicy, BufferOfDicts, BufferOfDictLists
from priority_policy_buffers import PPBCustom, PPBEventsPerObjectType, PPBLifespanPerObject, PPBLifespanPerObjectType, PPBObjectsPerEvent, PPBObjectsPerObjectType, PPBStridePerObject, PPBStridePerObjectType, PrioPolicyBuffer, PrioPolicyOrder
from utils import Event, O2OUpdate, ObjectAttributeUpdate, EventStream


class OcdfgBuffer(object):
    """
    Represents a streaming Object-Centric Directly-Follows Graph.
    The internal model buffers are maintained via a cache policy that is optionally combined with a priority policy.

    Attributes
    ----------
    node_buf_size : int
        Maximum number of buffered items in node buffer.
    arc_buf_size : int
        Maximum number of buffered items in arc buffer.
    cp : CachePolicy
        Cache policy used to replace node or arc buffer items when respective buffer is full.
    ot : str
        Object type to which all node and arc buffer items belong; i.e. a streaming DFG is maintained for the given object type. If None, all object types share a node and arc buffer.
    max_counter : int
        Maximum counter value (e.g. for key frequency or cache age) before reset.
    coupled_removal : bool
        If enabled, node and arc buffer are synchronized on the basis of object IDs (if `ot` is not None) or object types (if `ot` is None) after processing a stream item.
    pp_buf : PrioPolicyBuffer
        Optional priority-policy buffer that is maintained alongside node and arc buffer for removal decision based on additional object-centric characteristics.
    c_mon : CacheMonitor
        Optional CacheMonitor object for collecting information about cache behavior during stream processing; used for evaluation.
    rt_mon : RuntimeMonitor
        Optional RuntimeMonitor object for collecting information about runtime requirements during stream processing; used for evaluation.
    node_buf : BufferOfDicts
        Model buffer tracking information about in-coming activities.
    arc_buf : BufferOfDictLists
        Model buffer tracking information about directly-follows relations between activities.
    """
    
    def __init__(
            self, 
            node_buf_size : int, 
            arc_buf_size : int, 
            cp : CachePolicy, 
            ot : str = None, 
            max_counter : int = 100000, 
            coupled_removal : bool = False, 
            pp_buf : PrioPolicyBuffer = None, 
            cache_monitor : CacheMonitor = None, 
            runtime_monitor : RuntimeMonitor = None
        ):
        """
        Initializes an OcdfgBuffer object.

        Parameters
        ----------
        node_buf_size : int
            Maximum number of buffered items in node buffer.
        arc_buf_size : int
            Maximum number of buffered items in arc buffer.
        cp : CachePolicy
            Cache policy used to replace node or arc buffer items when respective buffer is full.
        ot : str, default=None
            Object type to which all node and arc buffer items belong; i.e. a streaming DFG is maintained for the given object type. If None, all object types share a node and arc buffer.
        max_counter : int, default=10000
            Maximum counter value (e.g. for key frequency or cache age) before reset.
        coupled_removal : bool, default=False
            If enabled, node and arc buffer are synchronized on the basis of object IDs (if `ot` is not None) or object types (if `ot` is None) after processing a stream item.
        pp_buf : PrioPolicyBuffer, default=None
            Optional priority-policy buffer that is maintained alongside node and arc buffer for removal decision based on additional object-centric characteristics.
        cache_monitor : CacheMonitor, default=None
            Optional CacheMonitor object for collecting information about cache behavior during stream processing; used for evaluation.
        runtime_monitor : RuntimeMonitor, default=None
            Optional RuntimeMonitor object for collecting information about runtime requirements during stream processing; used for evaluation.

        Raises
        ------
        AttributeError
            Error occurs if OC-DFG with buffers shared by object types is combined with object-based priority policy or vice-versa. Error also occurs when running cache and runtime monitor simultaneously.
        """
        self.node_buf_size = node_buf_size
        self.arc_buf_size = arc_buf_size
        self.cp = cp
        self.ot = ot                # if None, node and arc buffer contain several OTs
        self.max_counter = max_counter
        self.coupled_removal = coupled_removal
        
        # Check for allowed combinations of (OC-)DFG buffers and priority-policy buffers per object/OT
        if isinstance(pp_buf, (PPBStridePerObject, PPBLifespanPerObject)):
            if self.ot:
                self.pp_buf = pp_buf
            else:
                raise AttributeError(f'Combining mixed OC-DFG buffer and priority-policy buffer based on {pp_buf.pp.value} not feasible. Need priority policy on OT basis.')
        elif not isinstance(pp_buf, type(None)):
            if self.ot:
                raise AttributeError(f'Combining DFG buffer for OT {self.ot} and priority-policy buffer based on {pp_buf.pp.value} not feasible. Need priority policy on object basis.')
            else:
                self.pp_buf = pp_buf
        else:
            self.pp_buf = pp_buf
        
        if cache_monitor is not None and runtime_monitor is not None:
            raise AttributeError(f'Running CacheMonitor and RuntimeMonitor at the same time not recommended since CacheMonitor slows down computations! Pick one.')
        
        self.c_mon = cache_monitor
        # Register buffer sizes for cache monitor, if not already done externally
        if self.c_mon is not None and self.c_mon.total_model_buf_sizes is None:
            self.c_mon.set_total_buf_sizes(node_buf_size + arc_buf_size)
        self.rt_mon = runtime_monitor

        self.node_buf = BufferOfDicts(node_buf_size, cp, 'OC-DFG node buffer', NODE_BUF_KEY, ot, max_counter)
        self.arc_buf = BufferOfDictLists(arc_buf_size, cp, 'OC-DFG arc buffer', ARC_BUF_KEY, ot, max_counter)

    def process_stream(self, stream : list[Union[Event, O2OUpdate, ObjectAttributeUpdate]]) -> None:
        """
        Updates streaming representation by iterating over all stream items from start to finish.

        Parameters
        ----------
        stream : list[Union[Event, O2OUpdate, ObjectAttributeUpdate]]
            Object-centric event stream simulated as time-ordered list of stream items.

        Returns
        -------
        None
        """
        if self.c_mon is not None:
            self.c_mon.register_stream_size(stream)
        
        if self.rt_mon is not None:
            stream_start_time = time.time_ns()
        
        for stream_item in stream:
            if self.c_mon is not None:
                self.c_mon.increase_stream_item_count()
                if self.c_mon.check_for_ot_frac_per_buffer():
                    if self.ot is None:
                        for model_buf in [self.node_buf, self.arc_buf]:
                            self.c_mon.register_ot_frac_for_mixed_buffer(
                                model_buf,
                                nested_ot_key=OBJECT_TYPE, 
                                ot_tup_as_buf_key=False
                            )
                    else:
                        for model_buf in [self.node_buf, self.arc_buf]:
                            self.c_mon.register_ot_frac_for_single_ot_buffers(
                                {self.ot: model_buf}
                            )
            elif self.rt_mon is not None:
                self.rt_mon.increase_stream_item_count()

            if isinstance(stream_item, Event):
                if self.rt_mon is not None:
                    item_processing_start = time.time_ns()

                self.update_buffers(stream_item)
                self.clean_up_buffers()
            
                if self.rt_mon is not None:
                    self.rt_mon.register_item_processing_time(time.time_ns() - item_processing_start, stream_item.__class__.__name__)

        if self.rt_mon is not None:
            self.rt_mon.register_total_stream_processing_time(time.time_ns() - stream_start_time)

    def clean_up_buffers(self) -> None:
        """
        Synchronizes model buffers with each other in case of coupled removal, then synchronizes optional priority-policy buffer with model buffers.
        
        Returns
        -------
        None
        """
        if not isinstance(self.pp_buf, (type(None), PPBCustom)) or self.coupled_removal:
            # Monitor clean-up time
            if self.rt_mon is not None:
                clean_up_start_time = time.time_ns()

            # Perform OC-DFG buffer clean-up based on OT or OID and coupled removal if specified
            if self.ot is not None:
                node_buf_keys = set(self.node_buf.buf.keys())
                arc_buf_keys = self.arc_buf.get_buffered_nested_values_for_key(OID)
            else:
                node_buf_keys = self.node_buf.get_buffered_nested_values_for_key(OBJECT_TYPE)
                arc_buf_keys = self.arc_buf.get_buffered_nested_values_for_key(OBJECT_TYPE)
            
            if self.coupled_removal:
                model_buf_keys = node_buf_keys.intersection(arc_buf_keys)
                self.node_buf.reduce_to_oids_ots(model_buf_keys, nested_key=None if self.ot is not None else OBJECT_TYPE)
                self.arc_buf.reduce_to_oids_ots(model_buf_keys, nested_key=OID if self.ot is not None else OBJECT_TYPE)

                if not isinstance(self.pp_buf, (type(None), PPBCustom)):
                    self.pp_buf.clean_up(model_buf_keys)
            else:
                if not isinstance(self.pp_buf, (type(None), PPBCustom)):
                    self.pp_buf.clean_up(node_buf_keys.union(arc_buf_keys))

            # Register updated PPB size
            if self.c_mon is not None and not isinstance(self.pp_buf, type(None)):
                self.c_mon.update_ppb_size(len(self.pp_buf))
            
            # Register buffer-clean-up time
            if self.rt_mon is not None:
                self.rt_mon.register_buf_update_time(time.time_ns() - clean_up_start_time, 'OC-DFG buffer clean-up')
        else:
            return

    def update_buffers(self, stream_item : Event, external_oid_ot_to_pp_rank : dict[str, float] = None) -> None:
        """
        Update streaming OC-DFG for in-coming event.

        Parameters
        ----------
        stream_item : Event
            Event from stream for which streaming OC-DFG is updated.
        external_oid_ot_to_pp_rank : dict[str, float], default=None
            Optional mapping of object IDs or object types to priority-policy-based rank, otherwise determined during update if priority policy is used.

        Returns
        -------
        None
        """
        if isinstance(stream_item, Event):
            oid_ot_to_pp_rank = None
            if self.pp_buf is not None:
                if not isinstance(self.pp_buf, (PPBCustom)):
                    if self.rt_mon is not None:
                        ppb_start_time = time.time_ns()

                    self.pp_buf.update(stream_item)

                    if self.rt_mon is not None:
                        self.rt_mon.register_buf_update_time(time.time_ns() - ppb_start_time, self.pp_buf.buf_name)
                
                oid_ot_to_pp_rank = self.pp_buf.get_normalized_rank_by_pp()
            
            # Used for OCPN buffer, where OcdfgBuffer is re-used internally, but buffer-spanning priority-policy buffer is maintained externally inside OCPN buffer
            if external_oid_ot_to_pp_rank is not None:
                oid_ot_to_pp_rank = external_oid_ot_to_pp_rank

            curr_ts = stream_item.time
            curr_act = stream_item.activity
            curr_oids = list()

            # Filter by object type if buffer is aimed at single OT
            if self.ot:
                curr_oids = [(d['objectId'], self.ot) for d in stream_item.e2o_relations if d['objectType'] == self.ot]
            else:
                for d in stream_item.e2o_relations:
                    oid = d['objectId']
                    ot = d['objectType']
                    curr_oids.append((oid, ot))
            
            # Add adjusted target-activity frequency per OT per new arc
            curr_oids = [(oid, ot, 1/len(stream_item.e2o_relations)) for oid, ot in curr_oids]

            for new_oid, new_ot, new_target_freq in curr_oids:
                # Update arc buffer w/ new items for OID in current Event
                oid_dict = self.node_buf.buf[new_oid] if new_oid in self.node_buf.buf else None
                prev_act = oid_dict[ACTIVITY] if oid_dict else None
                act_dur = curr_ts - oid_dict[LAST_SEEN] if oid_dict else None
                
                new_arc_dict = {OID: new_oid, TARGET_ACTIVITY_FREQ: new_target_freq, ACTIVITY_DURATION: act_dur}
                if not self.ot:
                    new_arc_dict[OBJECT_TYPE] = new_ot
                
                if self.rt_mon is not None:
                    arc_buf_start_time = time.time_ns()

                self.update_arc_buf((prev_act, curr_act), new_arc_dict, oid_ot_to_pp_rank)

                if self.rt_mon is not None:
                    self.rt_mon.register_buf_update_time(time.time_ns() - arc_buf_start_time, self.arc_buf.buf_name)

                # Update node buffer w/ new items for OID after updating its arc accordingly
                node_dict_overwrite = {LAST_SEEN: curr_ts, ACTIVITY: curr_act}
                if self.ot:
                    node_dict_init = {}
                else:
                    # Save object type in case buffer is not aimed at one specific object type
                    # NOTE: cache age and frequency for LFU and LFU-DA can only be determined at time of insertion
                    node_dict_init = {OBJECT_TYPE: new_ot}

                if self.rt_mon is not None:
                    node_buf_start_time = time.time_ns()

                self.update_node_buf(new_oid, node_dict_overwrite, node_dict_init, oid_ot_to_pp_rank)

                if self.rt_mon is not None:
                    self.rt_mon.register_buf_update_time(time.time_ns() - node_buf_start_time, self.node_buf.buf_name)
        else:
            return

    def update_node_buf(self, buf_key : str, buf_dict_overwrite : dict[str, Any], buf_dict_init : dict[str, Any], oid_ot_to_pp_rank : dict[str, float] = None) -> None:
        """
        Insert new item into OC-DFG node buffer.

        Parameters
        ----------
        buf_key : str
            Key of new buffer item, i.e. object ID.
        buf_dict_overwrite : dict[str, Any]
            Key-value pairs of associated value dictionary that is inserted for given object ID.
        buf_dict_init : dict[str, Any]
            Key-value pairs to initialize in value dictionary once upon new insertion of object ID.
        oid_ot_to_pp_rank : dict[str, float], default=None
            Optional mapping of object IDs or object types to priority-policy-based rank.
        
        Returns
        -------
        None
        """
        if self.c_mon is not None:
            num_hits = 1 if buf_key in self.node_buf.buf else 0
            hit_miss_ot = buf_dict_init[OBJECT_TYPE] if self.ot is None else self.ot
            evicted_ot = None
        
        # Remove objects according to cache policy
        if self.node_buf.is_full() and buf_key not in self.node_buf.buf:
            if self.pp_buf is None:
                oid_pop = self.node_buf.get_removal_key_by_cp()
            else:
                oid_to_rank = dict()
                oid_to_cp_rank = self.node_buf.get_normalized_rank_by_cp()
                
                for oid_key in oid_to_cp_rank:
                    oid_ot_key = oid_key if self.ot else self.node_buf.buf[oid_key][OBJECT_TYPE]
                    if oid_ot_key in oid_ot_to_pp_rank:
                        oid_to_rank[oid_key] = oid_to_cp_rank[oid_key] + oid_ot_to_pp_rank[oid_ot_key]
                    else:
                        oid_to_rank[oid_key] = 2 * oid_to_cp_rank[oid_key]
            
                oid_pop = min(oid_to_rank.items(), key=lambda tup: tup[1])[0]
            
            if self.c_mon is not None:
                evicted_ot = self.node_buf.buf[oid_pop][OBJECT_TYPE] if self.ot is None else self.ot

            self.node_buf.remove_item(oid_pop)
        
        # Insert/update buffer item for OID
        self.node_buf.insert_new_item_by_cp(buf_key, buf_dict_overwrite, buf_dict_init)

        if self.c_mon is not None:
            self.c_mon.register_buf_insertion(num_hits, hit_miss_ot, self.node_buf.buf_name)

            if evicted_ot is not None:
                full_eviction = evicted_ot not in self.node_buf.get_buffered_nested_values_for_key(OBJECT_TYPE) if self.ot is None else False
                self.c_mon.register_buf_eviction(evicted_ot, full_eviction, self.node_buf.buf_name)
    
    def update_arc_buf(self, buf_key : Tuple[str, str], buf_dict : dict[str, Any], oid_ot_to_pp_rank : dict[str, float] = None) -> None:
        """
        Insert new item into OC-DFG arc buffer.

        Parameters
        ----------
        buf_key : Tuple[str, str]
            Key of new buffer item, i.e. activity-activity pair.
        buf_dict : dict[str, Any]
            Key-value pairs of associated value dictionary that is inserted for given arc.
        oid_ot_to_pp_rank : dict[str, float], default=None
            Optional mapping of object IDs or object types to priority-policy-based rank.
        
        Returns
        -------
        None
        """
        oid_ot_nested_key = OID if self.ot else OBJECT_TYPE

        if self.c_mon is not None:
            num_hits = 1 if buf_key in self.arc_buf.buf else 0
            hit_miss_ot = buf_dict[OBJECT_TYPE] if self.ot is None else self.ot
            evicted_ot = None
        
        # Remove arcs according to cache policy
        # NOTE: no need to additionally check "new_arc not in self.arc_buf.buf", since each arc creates new OID-related item in list associated w/ arc
        if self.arc_buf.is_full():
            if self.pp_buf is None:
                arc_pop_key, arc_pop_idx = self.arc_buf.get_removal_key_by_cp()
            else:
                key_idx_to_rank = dict()
                key_idx_to_cp_rank = self.arc_buf.get_normalized_rank_by_cp()
                
                for arc_key, arc_list_id in key_idx_to_cp_rank:
                    oid_ot_key = self.arc_buf.buf[arc_key][arc_list_id][oid_ot_nested_key]
                    if oid_ot_key in oid_ot_to_pp_rank:
                        key_idx_to_rank[(arc_key, arc_list_id)] = key_idx_to_cp_rank[(arc_key, arc_list_id)] + oid_ot_to_pp_rank[oid_ot_key]
                    else:
                        key_idx_to_rank[(arc_key, arc_list_id)] = 2 * key_idx_to_cp_rank[(arc_key, arc_list_id)]
                
                arc_pop_key, arc_pop_idx = min(key_idx_to_rank.items(), key=lambda tup: tup[1])[0]
            
            if self.c_mon is not None:
                evicted_ot = self.arc_buf.buf[arc_pop_key][arc_pop_idx][OBJECT_TYPE] if self.ot is None else self.ot

            self.arc_buf.remove_item(arc_pop_key, arc_pop_idx)
            
        # Insert/update buffer item for arc
        self.arc_buf.insert_new_item_by_cp(buf_key, buf_dict)

        if self.c_mon is not None:
            self.c_mon.register_buf_insertion(num_hits, hit_miss_ot, self.arc_buf.buf_name)

            if evicted_ot is not None:
                full_eviction = evicted_ot not in self.arc_buf.get_buffered_nested_values_for_key(OBJECT_TYPE) if self.ot is None else False
                self.c_mon.register_buf_eviction(evicted_ot, full_eviction, self.arc_buf.buf_name)
    
    def create_monitor_dataframes(self) -> None:
        """
        Turns information collected by runtime or cache monitor, if enabled, into DataFrames.

        Returns
        -------
        None
        """
        if self.rt_mon is not None:
            self.rt_mon.create_dataframes()
        elif self.c_mon is not None:
            self.c_mon.create_dataframes()
        else:
            return

    def __str__(self) -> str:
        """
        Creates string describing streaming OC-DFG including its parameters (cache policy etc.) and items buffered in each model buffer in tabular format.

        Returns
        -------
        str
        """
        node_buf_cols = [NODE_BUF_KEY, ACTIVITY, LAST_SEEN]
        arc_buf_cols = [ARC_BUF_KEY, OID, TARGET_ACTIVITY_FREQ, ACTIVITY_DURATION]

        if not self.ot:
            node_buf_cols.insert(1, OBJECT_TYPE)
            arc_buf_cols.insert(2, OBJECT_TYPE)

        if self.cp in [CachePolicy.LFU, CachePolicy.LFU_DA]:
            node_buf_cols.append(FREQUENCY)
            arc_buf_cols.append(FREQUENCY)

        if self.cp == CachePolicy.LFU_DA:
            node_buf_cols.append(CACHE_AGE)
            arc_buf_cols.append(CACHE_AGE)

        ret = f'Coupled removal for buffered OC-DFG model: {self.coupled_removal}\n'
        ret += self.node_buf.to_string(node_buf_cols) + self.arc_buf.to_string(arc_buf_cols)
        if self.pp_buf is not None:
            ret += self.pp_buf.to_string()
        
        return ret


class OcdfgBufferPerObjectType(object):
    """
    Represents a streaming Object-Centric Directly-Follows Graph with separate model buffers per object type.
    The internal model buffers are maintained via a cache policy that is optionally combined with a priority policy.

    Attributes
    ----------
    node_buf_size : int
        Maximum number of buffered items in each node buffer.
    arc_buf_size : int
        Maximum number of buffered items in each arc buffer.
    cp : CachePolicy
        Cache policy used to replace node or arc buffer items when respective buffer is full.
    max_counter : int
        Maximum counter value (e.g. for key frequency or cache age) before reset.
    coupled_removal : bool
        If enabled, node and arc buffer are synchronized on the basis of object IDs after processing a stream item.
    pp_buf : PrioPolicyBuffer
        Optional object-based priority-policy buffer that is maintained alongside node and arc buffers for removal decision based on additional object-centric characteristics.
    c_mon : CacheMonitor
        Optional CacheMonitor object for collecting information about cache behavior during stream processing; used for evaluation.
    rt_mon : RuntimeMonitor
        Optional RuntimeMonitor object for collecting information about runtime requirements during stream processing; used for evaluation.
    dfg_bufs : dict[str, OcdfgBuffer]
        Mapping of object types to streaming DFGs.
    """

    def __init__(
            self, 
            node_buf_size : int, 
            arc_buf_size : int, 
            cp : CachePolicy, 
            max_counter : int = 10000, 
            coupled_removal : bool = False, 
            pp_buf : PrioPolicyBuffer = None, 
            cache_monitor : CacheMonitor = None, 
            runtime_monitor : RuntimeMonitor = None
        ):
        """
        Initializes an OcdfgBufferPerObjectType object.

        Parameters
        ----------
        node_buf_size : int
            Maximum number of buffered items in each node buffer.
        arc_buf_size : int
            Maximum number of buffered items in each arc buffer.
        cp : CachePolicy
            Cache policy used to replace node or arc buffer items when respective buffer is full.
        max_counter : int, default=10000
            Maximum counter value (e.g. for key frequency or cache age) before reset.
        coupled_removal : bool, default=False
            If enabled, node and arc buffer are synchronized on the basis of object IDs after processing a stream item.
        pp_buf : PrioPolicyBuffer, default=None
            Optional priority-policy buffer that is maintained alongside node and arc buffer for removal decision based on additional object-centric characteristics.
        cache_monitor : CacheMonitor, default=None
            Optional CacheMonitor object for collecting information about cache behavior during stream processing; used for evaluation.
        runtime_monitor : RuntimeMonitor, default=None
            Optional RuntimeMonitor object for collecting information about runtime requirements during stream processing; used for evaluation.

        Raises
        ------
        AttributeError
            Error occurs if OC-DFG with separate model buffers per object type is combined with type-based priority policy. Error also occurs when running cache and runtime monitor simultaneously.
        """
        self.node_buf_size = node_buf_size
        self.arc_buf_size = arc_buf_size
        self.cp = cp
        self.max_counter = max_counter
        self.coupled_removal = coupled_removal
        
        # Check for allowed combinations of DFG buffers and priority-policy buffers
        if isinstance(pp_buf, (PPBStridePerObject, PPBLifespanPerObject)):
            self.pp_buf = pp_buf
        elif not isinstance(pp_buf, type(None)):
            raise AttributeError(f'Combining OC-DFG buffer per OT and priority-policy buffer based on {pp_buf.pp.value} not feasible. Need priority policy on object basis.')
        else:
            self.pp_buf = pp_buf

        if cache_monitor is not None and runtime_monitor is not None:
            raise AttributeError(f'Running CacheMonitor and RuntimeMonitor at the same time not recommended since CacheMonitor slows down computations! Pick one.')
        self.c_mon = cache_monitor
        self.rt_mon = runtime_monitor

        self.dfg_bufs = dict()
    
    def process_stream(self, stream : list[Union[Event, O2OUpdate, ObjectAttributeUpdate]]) -> None:
        """
        Updates streaming representation by iterating over all stream items from start to finish.

        Parameters
        ----------
        stream : list[Union[Event, O2OUpdate, ObjectAttributeUpdate]]
            Object-centric event stream simulated as time-ordered list of stream items.

        Returns
        -------
        None
        """
        if self.c_mon is not None:
            self.c_mon.register_stream_size(stream)

        if self.rt_mon is not None:
            stream_start_time = time.time_ns()

        for stream_item in stream:
            if self.c_mon is not None:
                self.c_mon.increase_stream_item_count()
                if self.c_mon.check_for_ot_frac_per_buffer():
                    self.c_mon.register_ot_frac_for_single_ot_buffers({ot: dfg.arc_buf for ot, dfg in self.dfg_bufs.items()})
                    self.c_mon.register_ot_frac_for_single_ot_buffers({ot: dfg.node_buf for ot, dfg in self.dfg_bufs.items()})
            elif self.rt_mon is not None:
                self.rt_mon.increase_stream_item_count()

            if isinstance(stream_item, Event):
                if self.rt_mon is not None:
                    item_processing_start = time.time_ns()

                # Update/create buffered DFG models for involved object types
                curr_ots = set([d['objectType'] for d in stream_item.e2o_relations])
                for ot in curr_ots:
                    if ot not in self.dfg_bufs:
                        self.dfg_bufs[ot] = OcdfgBuffer(
                            self.node_buf_size, 
                            self.arc_buf_size, 
                            self.cp, 
                            ot, 
                            self.max_counter, 
                            self.coupled_removal, 
                            deepcopy(self.pp_buf), 
                            cache_monitor=self.c_mon, 
                            runtime_monitor=self.rt_mon
                        )

                    self.dfg_bufs[ot].update_buffers(stream_item)
                    self.dfg_bufs[ot].clean_up_buffers()

                if self.rt_mon is not None:
                    self.rt_mon.register_item_processing_time(time.time_ns() - item_processing_start, stream_item.__class__.__name__)
        
        if self.rt_mon is not None:
            self.rt_mon.register_total_stream_processing_time(time.time_ns() - stream_start_time)
        
        if self.c_mon is not None:
            self.c_mon.set_total_buf_sizes((self.arc_buf_size + self.node_buf_size) * len(self.dfg_bufs))

    def create_monitor_dataframes(self) -> None:
        """
        Turns information collected by runtime or cache monitor, if enabled, into DataFrames.

        Returns
        -------
        None
        """
        if self.rt_mon is not None:
            self.rt_mon.create_dataframes()
        elif self.c_mon is not None:
            self.c_mon.create_dataframes()
        else:
            return

    def __str__(self) -> str:
        """
        Creates string describing streaming OC-DFG including its parameters (cache policy etc.) and items buffered in each DFG model buffer in tabular format.

        Returns
        -------
        str
        """
        ret = f'Coupled removal of buffered DFG-per-OT model: {self.coupled_removal}\n'
        for ot in self.dfg_bufs:
            ret += str(self.dfg_bufs[ot])
        return ret


class TotemBuffer(object):
    """
    Represents a streaming Temporal Object Type Model.
    The internal model buffers are maintained via a cache policy that is optionally combined with a priority policy.

    Attributes
    ----------
    tr_buf_size : int
        Maximum number of buffered items in temporal-relation buffer.
    ec_buf_size : int
        Maximum number of buffered items in event-cardinality buffer.
    lc_buf_size : int
        Maximum number of buffered items in log-cardinality buffer.
    cp : CachePolicy
        Cache policy used to replace model-buffer items when respective buffer is full.
    max_counter : int
        Maximum counter value (e.g. for key frequency or cache age) before reset.
    coupled_removal : bool
        If enabled, model buffers are synchronized on the basis of object types after processing a stream item.
    pp_buf : PrioPolicyBuffer
        Optional priority-policy buffer that is maintained alongside model buffers for removal decision based on additional object-centric characteristics.
    c_mon : CacheMonitor
        Optional CacheMonitor object for collecting information about cache behavior during stream processing; used for evaluation.
    rt_mon : RuntimeMonitor
        Optional RuntimeMonitor object for collecting information about runtime requirements during stream processing; used for evaluation.
    tr_buf : BufferOfDicts
        Model buffer tracking information about lifespan intervals of objects.
    ec_buf : BufferOfDictLists
        Model buffer tracking information about event cardinalities.
    lc_buf : BufferOfDictLists
        Model buffer tracking information about log cardinalities.
    """
    
    def __init__(
            self, 
            tr_buf_size : int, 
            ec_buf_size : int, 
            lc_buf_size : int, 
            cp : CachePolicy, 
            max_counter : int = 10000, 
            coupled_removal : bool = False, 
            pp_buf : PrioPolicyBuffer = None, 
            cache_monitor : CacheMonitor = None, 
            runtime_monitor : RuntimeMonitor = None
        ):
        """
        Initializes a TotemBuffer object.

        Parameters
        ----------
        tr_buf_size : int
            Maximum number of buffered items in temporal-relation buffer.
        ec_buf_size : int
            Maximum number of buffered items in event-cardinality buffer.
        lc_buf_size : int
            Maximum number of buffered items in log-cardinality buffer.
        cp : CachePolicy
            Cache policy used to replace model-buffer items when respective buffer is full.
        max_counter : int, default=10000
            Maximum counter value (e.g. for key frequency or cache age) before reset.
        coupled_removal : bool, default=False
            If enabled, model buffers are synchronized on the basis of object types after processing a stream item.
        pp_buf : PrioPolicyBuffer, default=None
            Optional priority-policy buffer that is maintained alongside model buffers for removal decision based on additional object-centric characteristics.
        cache_monitor : CacheMonitor, default=None
            Optional CacheMonitor object for collecting information about cache behavior during stream processing; used for evaluation.
        runtime_monitor : RuntimeMonitor, default=None
            Optional RuntimeMonitor object for collecting runtime measurements during stream processing; used for evaluation.

        Raises
        ------
        AttributeError
            Error occurs if streaming TOTeM is combined with object-based priority policy. Error also occurs when running cache and runtime monitor simultaneously.
        """
        self.tr_buf_size = tr_buf_size
        self.ec_buf_size = ec_buf_size
        self.lc_buf_size = lc_buf_size
        self.cp = cp
        self.max_counter = max_counter
        self.coupled_removal = coupled_removal

        # Check for allowed combinations of TOTeM buffers and priority-policy buffers per OT
        if isinstance(pp_buf, (PPBStridePerObject, PPBLifespanPerObject)):
            raise AttributeError(f'Combining mixed-OT TOTeM buffer and priority-policy buffer based on {pp_buf.pp.value} not feasible. Need priority policy on OT basis.')
        else:
            self.pp_buf = pp_buf
        
        if cache_monitor is not None and runtime_monitor is not None:
            raise AttributeError(f'Running CacheMonitor and RuntimeMonitor at the same time not recommended since CacheMonitor slows down computations! Pick one.')
        
        self.c_mon = cache_monitor
        if self.c_mon is not None and self.c_mon.total_model_buf_sizes is None:
            self.c_mon.set_total_buf_sizes(tr_buf_size + ec_buf_size + lc_buf_size)
        self.rt_mon = runtime_monitor

        self.tr_buf = BufferOfDicts(tr_buf_size, cp, 'TOTeM TR buffer', TR_BUF_KEY, max_counter=max_counter)
        self.lc_buf = BufferOfDictLists(lc_buf_size, cp, 'TOTeM LC buffer', LC_BUF_KEY, max_counter=max_counter)
        self.ec_buf = BufferOfDictLists(ec_buf_size, cp, 'TOTeM EC buffer', EC_BUF_KEY, max_counter=max_counter)

    def process_stream(self, stream : list[Union[Event, O2OUpdate, ObjectAttributeUpdate]], enrich_o2o : bool = False) -> None:
        """
        Updates streaming representation by iterating over all stream items from start to finish.

        Parameters
        ----------
        stream : list[Union[Event, O2OUpdate, ObjectAttributeUpdate]]
            Object-centric event stream simulated as time-ordered list of stream items.
        enrich_o2o : bool, default=False
            If enabled, object-to-object updates/relations are derived from objects involved in an in-coming event are re-inserted at the front of the stream.

        Returns
        -------
        None
        """
        if self.c_mon is not None:
            self.c_mon.register_stream_size(stream)
        
        if self.rt_mon is not None:
            stream_start_time = time.time_ns()
        
        i = 0
        while i < len(stream):
            if self.c_mon is not None:
                self.c_mon.increase_stream_item_count()
                if self.c_mon.check_for_ot_frac_per_buffer():
                    self.c_mon.register_ot_frac_for_mixed_buffer(self.tr_buf, nested_ot_key=OBJECT_TYPE)
                    self.c_mon.register_ot_frac_for_mixed_buffer(self.ec_buf, ot_tup_as_buf_key=True)
                    self.c_mon.register_ot_frac_for_mixed_buffer(self.lc_buf, ot_tup_as_buf_key=True)
            elif self.rt_mon is not None:
                self.rt_mon.increase_stream_item_count()

            # Enrich stream w/ O2OUpdates derived from Event if specified
            stream_item = stream[i]
            if enrich_o2o and isinstance(stream_item, Event):
                e2o_tups = [(d['objectId'], d['objectType']) for d in stream_item.e2o_relations]
                new_o2o_rels = itertools.product(e2o_tups, e2o_tups)
                j = 1
                for (src_oid, src_type), (target_oid, target_type) in new_o2o_rels:
                    if src_oid == target_oid or target_type is None or src_type is None or src_type == target_type:
                        # Skip enriched O2O relations where target type matches source type since these are not considered in temporal relations anyway
                        continue
                    
                    stream.insert(
                        i + j,
                        O2OUpdate(
                            time=stream_item.time,
                            id=src_oid,
                            type=src_type,
                            target_id=target_oid,
                            target_type=target_type,
                            # Set event activity name as qualifier of derived O2O relation
                            qualifier=stream_item.activity
                        )
                    )
                    j += 1

            if isinstance(stream_item, (Event, O2OUpdate)):
                if self.rt_mon is not None:
                    item_processing_start_time = time.time_ns()
                
                self.update_buffers(stream_item)
                self.clean_up_buffers()
            
                if self.rt_mon is not None:
                    self.rt_mon.register_item_processing_time(time.time_ns() - item_processing_start_time, stream_item.__class__.__name__)
            
            i += 1
        
        if self.rt_mon is not None:
            self.rt_mon.register_total_stream_processing_time(time.time_ns() - stream_start_time)
    
    def clean_up_buffers(self) -> None:
        """
        Synchronizes model buffers with each other in case of coupled removal, then synchronizes optional priority-policy buffer with model buffers.
        
        Returns
        -------
        None
        """
        if not isinstance(self.pp_buf, (type(None), PPBCustom)) or self.coupled_removal:
            # Monitor time it takes to perform a clean-up
            if self.rt_mon is not None:
                clean_up_start_time = time.time_ns()
        
            # Perform buffer clean-up including coupled removal if specified
            tr_buf_ots = self.tr_buf.get_buffered_nested_values_for_key(OBJECT_TYPE)
            ec_buf_ots = set(itertools.chain.from_iterable(self.ec_buf.buf.keys()))
            lc_buf_ots = set(itertools.chain.from_iterable(self.lc_buf.buf.keys()))
            
            if self.coupled_removal:
                model_buf_ots = tr_buf_ots.intersection(ec_buf_ots).intersection(lc_buf_ots)
                self.tr_buf.reduce_to_oids_ots(model_buf_ots, nested_key=OBJECT_TYPE)
                self.ec_buf.reduce_to_oids_ots(model_buf_ots, key_is_tuple=True)
                self.lc_buf.reduce_to_oids_ots(model_buf_ots, key_is_tuple=True)

                if not isinstance(self.pp_buf, (type(None), PPBCustom)):
                    self.pp_buf.clean_up(model_buf_ots)
            else:
                if not isinstance(self.pp_buf, (type(None), PPBCustom)):
                    self.pp_buf.clean_up(tr_buf_ots.union(ec_buf_ots).union(lc_buf_ots))
            
            # Monitor updates PPB size
            if self.c_mon is not None and not isinstance(self.pp_buf, type(None)):
                self.c_mon.update_ppb_size(len(self.pp_buf))
            
            # Register buffer-clean-up time
            if self.rt_mon is not None:
                self.rt_mon.register_buf_update_time(time.time_ns() - clean_up_start_time, 'TOTeM buffer clean-up')
        else:
            return

    def update_buffers(self, stream_item : Union[Event, O2OUpdate]) -> None:
        """
        Update streaming TOTeM for in-coming event or object-to-object update.

        Parameters
        ----------
        stream_item : Union[Event, O2OUpdate]
            Stream item for which streaming TOTeM is updated.

        Returns
        -------
        None
        """
        # Reduce passes over E2O relations by writing new buffer items to TR, LC, EC buffer immediately instead of pre-defining new buffer items to iterate over each
        ot_to_pp_rank = None
        if self.pp_buf is not None:
            if isinstance(self.pp_buf, (PPBEventsPerObjectType, PPBObjectsPerEvent, PPBLifespanPerObjectType, PPBStridePerObjectType)) and isinstance(stream_item, Event) or isinstance(self.pp_buf, PPBObjectsPerObjectType) and isinstance(stream_item, (Event, O2OUpdate)):
                if self.rt_mon is not None:
                    ppb_start_time = time.time_ns()

                self.pp_buf.update(stream_item)

                if self.rt_mon is not None:
                    self.rt_mon.register_buf_update_time(time.time_ns() - ppb_start_time, self.pp_buf.buf_name)

            ot_to_pp_rank = self.pp_buf.get_normalized_rank_by_pp()

        curr_ts = stream_item.time
        ec_buf_unique_event_id = None
        ots_to_oids = dict()
        if isinstance(stream_item, Event):
            # Create OT-to-OIDs map for deriving new EC- and LC-buffer items
            for e2o_dict in stream_item.e2o_relations:
                ot = e2o_dict['objectType']
                oid = e2o_dict['objectId']
                ots_to_oids.setdefault(ot, list())
                ots_to_oids[ot].append(oid)

                # Define new items for TR buffer
                if self.rt_mon is not None:
                    tr_start_time = time.time_ns()

                self.update_tr_buf(oid, {LAST_SEEN: curr_ts}, {OBJECT_TYPE: ot, FIRST_SEEN: curr_ts}, ot_to_pp_rank)

                if self.rt_mon is not None:
                    self.rt_mon.register_buf_update_time(time.time_ns() - tr_start_time, self.tr_buf.buf_name)
        elif isinstance(stream_item, O2OUpdate):
            for ot, oid in [(stream_item.type, stream_item.id), (stream_item.target_type, stream_item.target_id)]:
                # Create OT-to-OID map to derive new LC-buffer items
                ots_to_oids.setdefault(ot, list())
                ots_to_oids[ot].append(oid)
        else:
            return
        
        # Define new items for LC and EC buffers
        if len(ots_to_oids) == 1:
            ot = next(iter(ots_to_oids))

            # Add new LC buffer item for all OIDs in case of just 1 OT in current Event
            for oid in itertools.chain.from_iterable(ots_to_oids.values()):
                if self.rt_mon is not None:
                    lc_start_time = time.time_ns()

                self.update_lc_buf((ot, None), {OBJECT_PAIR: (oid, None)}, ot_to_pp_rank)

                if self.rt_mon is not None:
                    self.rt_mon.register_buf_update_time(time.time_ns() - lc_start_time, self.lc_buf.buf_name)

            # Add new EC buffer item for only OT and if stream item is Event
            if isinstance(stream_item, Event):
                if self.rt_mon is not None:
                    ec_start_time = time.time_ns()

                ec_buf_unique_event_id = self.update_ec_buf((ot, None), {EVENT_CARD: [EC_ZERO, EC_ZERO_ONE, EC_ZERO_MANY]}, ec_buf_unique_event_id, ot_to_pp_rank)

                if self.rt_mon is not None:
                    self.rt_mon.register_buf_update_time(time.time_ns() - ec_start_time, self.ec_buf.buf_name)
        
        elif len(ots_to_oids) > 1:
            curr_ots = list(ots_to_oids.keys())
            for idx_ot_a in range(len(ots_to_oids)):
                ot_a = curr_ots[idx_ot_a]

                for idx_ot_b in range(idx_ot_a+1, len(curr_ots)):
                    ot_b = curr_ots[idx_ot_b]

                    if ot_a == ot_b:
                        continue

                    # Add new LC buffer items for each unique pair of objects of different OTs, sorted alphabetically
                    ot_pair_sorted = tuple(sorted((ot_a, ot_b)))
                    ot_src = ot_pair_sorted[0]
                    ot_target = ot_pair_sorted[1]
                    for oid_src in ots_to_oids[ot_src]:
                        for oid_target in ots_to_oids[ot_target]:
                            if self.rt_mon is not None:
                                lc_start_time = time.time_ns()

                            self.update_lc_buf((ot_src, ot_target), {OBJECT_PAIR: (oid_src, oid_target)}, ot_to_pp_rank)

                            if self.rt_mon is not None:
                                self.rt_mon.register_buf_update_time(time.time_ns() - lc_start_time, self.lc_buf.buf_name)

                    # Add EC buffer items if stream item is Event based on cardinalities in given direction and reverse direction of OT pair of different types
                    if isinstance(stream_item, Event):
                        for ot_src, ot_target in [(ot_a, ot_b), (ot_b, ot_a)]:
                            if len(ots_to_oids[ot_target]) == 1:
                                event_card = [EC_ONE, EC_ZERO_ONE, EC_ONE_MANY, EC_ZERO_MANY]
                            elif len(ots_to_oids[ot_target]) > 1:
                                event_card = [EC_ONE_MANY, EC_ZERO_MANY]
                            else:
                                raise ValueError(f'Cannot parse event cardinality for OT pair {(ot_src, ot_target)} in E2O {ots_to_oids}.')
                            if self.rt_mon is not None:
                                ec_start_time = time.time_ns()

                            ec_buf_unique_event_id = self.update_ec_buf((ot_src, ot_target), {EVENT_CARD: event_card}, ec_buf_unique_event_id, ot_to_pp_rank)

                            if self.rt_mon is not None:
                                self.rt_mon.register_buf_update_time(time.time_ns() - ec_start_time, self.ec_buf.buf_name)
        
        else:
            return

    def update_tr_buf(self, buf_key : str, buf_dict_overwrite : dict[str, Any], buf_dict_init : dict[str, Any], ot_to_pp_rank : dict[str, float] = None) -> None:
        """
        Insert new item into TOTeM temporal-relation buffer.

        Parameters
        ----------
        buf_key : str
            Key of new buffer item, i.e. object ID.
        buf_dict_overwrite : dict[str, Any]
            Key-value pairs of associated value dictionary that is inserted for given object ID.
        buf_dict_init : dict[str, Any]
            Key-value pairs to initialize in value dictionary once upon new insertion of object ID.
        ot_to_pp_rank : dict[str, float], default=None
            Optional mapping of object types to priority-policy-based rank.
        
        Returns
        -------
        None
        """
        if self.c_mon is not None:
            num_hits = 1 if buf_key in self.tr_buf.buf else 0
            hit_miss_ot = buf_dict_init[OBJECT_TYPE]
            evicted_ot = None

        # Remove objects according to cache policy
        if self.tr_buf.is_full() and buf_key not in self.tr_buf.buf:
            if ot_to_pp_rank is None:
                oid_pop = self.tr_buf.get_removal_key_by_cp()
            else:
                oid_to_cp_rank = self.tr_buf.get_normalized_rank_by_cp()
                oid_to_rank = dict()

                for oid_key in oid_to_cp_rank:
                    ot_key = self.tr_buf.buf[oid_key][OBJECT_TYPE]
                    if ot_key in ot_to_pp_rank:
                        oid_to_rank[oid_key] = oid_to_cp_rank[oid_key] + ot_to_pp_rank[ot_key]
                    else:
                        oid_to_rank[oid_key] = 2 * oid_to_cp_rank[oid_key]

                oid_pop = min(oid_to_rank.items(), key=lambda tup: tup[1])[0]
            
            if self.c_mon is not None:
                evicted_ot = self.tr_buf.buf[oid_pop][OBJECT_TYPE]

            self.tr_buf.remove_item(oid_pop)

        # Insert/update buffer item for OID
        self.tr_buf.insert_new_item_by_cp(buf_key, buf_dict_overwrite, buf_dict_init)

        if self.c_mon is not None:
            self.c_mon.register_buf_insertion(num_hits, hit_miss_ot, self.tr_buf.buf_name)

            if evicted_ot is not None:
                full_eviction = evicted_ot not in self.tr_buf.get_buffered_nested_values_for_key(OBJECT_TYPE)
                self.c_mon.register_buf_eviction(evicted_ot, full_eviction, self.tr_buf.buf_name)
    
    def update_lc_buf(self, buf_key : Tuple[str, str], buf_dict : dict[str, Any], ot_to_pp_rank : dict[str, float] = None) -> None:
        """
        Insert new item into TOTeM log-cardinality buffer.

        Parameters
        ----------
        buf_key : Tuple[str, str]
            Key of new buffer item, i.e. pair of object types.
        buf_dict : dict[str, Any]
            Key-value pairs of associated value dictionary that is inserted for given type pair.
        ot_to_pp_rank : dict[str, float], default=None
            Optional mapping of object types to priority-policy-based rank.
        
        Returns
        -------
        None
        """
        if self.c_mon is not None:
            num_hits = 1 if buf_key in self.lc_buf.buf else 0
            hit_miss_ots = set([buf_key[0], buf_key[1]])
            hit_miss_ots.discard(None)
            evicted_ots = set()
        
        # Remove LC buffer items according to cache policy
        if self.lc_buf.is_full():
            if ot_to_pp_rank is None:
                lc_pop_key, lc_pop_idx = self.lc_buf.get_removal_key_by_cp()
            else:
                key_idx_to_rank = dict()
                key_idx_to_cp_rank = self.lc_buf.get_normalized_rank_by_cp()

                for ot_pair_key, list_id in key_idx_to_cp_rank:
                    key_idx_to_rank[(ot_pair_key, list_id)] = 2 * key_idx_to_cp_rank[(ot_pair_key, list_id)]

                    if ot_pair_key[0] in ot_to_pp_rank and ot_pair_key[1] is not None and ot_pair_key[1] in ot_to_pp_rank:
                        key_idx_to_rank[(ot_pair_key, list_id)] += (ot_to_pp_rank[ot_pair_key[0]] + ot_to_pp_rank[ot_pair_key[1]])
                    elif ot_pair_key[0] in ot_to_pp_rank and (ot_pair_key[1] is None or ot_pair_key[1] not in ot_to_pp_rank):
                        key_idx_to_rank[(ot_pair_key, list_id)] += 2 * ot_to_pp_rank[ot_pair_key[0]]
                    elif ot_pair_key[0] not in ot_to_pp_rank and ot_pair_key[1] is not None and ot_pair_key[1] in ot_to_pp_rank:
                        key_idx_to_rank[(ot_pair_key, list_id)] += 2 * ot_to_pp_rank[ot_pair_key[1]]
                    else:
                        key_idx_to_rank[(ot_pair_key, list_id)] += 2 * key_idx_to_cp_rank[(ot_pair_key, list_id)]
                
                lc_pop_key, lc_pop_idx = min(key_idx_to_rank.items(), key=lambda tup: tup[1])[0]

            if self.c_mon is not None:
                evicted_ots.add(lc_pop_key[0])
                evicted_ots.add(lc_pop_key[1])
                evicted_ots.discard(None)
            
            self.lc_buf.remove_item(lc_pop_key, lc_pop_idx)
            
        # Insert/update buffer item for LC
        self.lc_buf.insert_new_item_by_cp(buf_key, buf_dict)

        if self.c_mon is not None:
            for hit_miss_ot in hit_miss_ots:
                self.c_mon.register_buf_insertion(num_hits, hit_miss_ot, self.lc_buf.buf_name)

            if len(evicted_ots) > 0:
                buffered_ots = set()
                for lc_buf_key in self.lc_buf.buf:
                    buffered_ots.add(lc_buf_key[0])
                    buffered_ots.add(lc_buf_key[1])
                buffered_ots.discard(None)

                for evicted_ot in evicted_ots:
                    full_eviction = evicted_ot not in buffered_ots
                    self.c_mon.register_buf_eviction(evicted_ot, full_eviction, self.lc_buf.buf_name)

    def update_ec_buf(self, buf_key : Tuple[str, str], buf_dict : dict[str, Any], unique_event_id : int = None, ot_to_pp_rank : dict[str, float] = None) -> int:
        """
        Insert new item into TOTeM log-cardinality buffer.

        Parameters
        ----------
        buf_key : Tuple[str, str]
            Key of new buffer item, i.e. pair of object types.
        buf_dict : dict[str, Any]
            Key-value pairs of associated value dictionary that is inserted for given type pair.
        unique_event_id : int, default=None
            Event ID that is unique w.r.t. IDs already buffered that uniquely identifies event associated with new buffer item.
        ot_to_pp_rank : dict[str, float], default=None
            Optional mapping of object types to priority-policy-based rank.
        
        Returns
        -------
        int
            Event ID that uniquely identifies event associated with new buffer item among already buffered events.
        """
        if self.c_mon is not None:
            num_hits = 1 if buf_key in self.ec_buf.buf else 0
            hit_miss_ots = set([buf_key[0], buf_key[1]])
            hit_miss_ots.discard(None)
            evicted_ots = set()

        # Remove EC items associated w/ OT pairs according to cache policy
        if self.ec_buf.is_full():
            if ot_to_pp_rank is None:
                ec_pop_key, ec_pop_idx = self.ec_buf.get_removal_key_by_cp()
            else:
                key_idx_to_rank = dict()
                key_idx_to_cp_rank = self.ec_buf.get_normalized_rank_by_cp()

                for ot_pair_key, list_id in key_idx_to_cp_rank:
                    key_idx_to_rank[(ot_pair_key, list_id)] = 2 * key_idx_to_cp_rank[(ot_pair_key, list_id)]

                    if ot_pair_key[0] in ot_to_pp_rank and ot_pair_key[1] is not None and ot_pair_key[1] in ot_to_pp_rank:
                        key_idx_to_rank[(ot_pair_key, list_id)] += (ot_to_pp_rank[ot_pair_key[0]] + ot_to_pp_rank[ot_pair_key[1]])
                    elif ot_pair_key[0] in ot_to_pp_rank and (ot_pair_key[1] is None or ot_pair_key[1] not in ot_to_pp_rank):
                        key_idx_to_rank[(ot_pair_key, list_id)] += 2 * ot_to_pp_rank[ot_pair_key[0]]
                    elif ot_pair_key[0] not in ot_to_pp_rank and ot_pair_key[1] is not None and ot_pair_key[1] in ot_to_pp_rank:
                        key_idx_to_rank[(ot_pair_key, list_id)] += 2 * ot_to_pp_rank[ot_pair_key[1]]
                    else:
                        key_idx_to_rank[(ot_pair_key, list_id)] += 2 * key_idx_to_cp_rank[(ot_pair_key, list_id)]
                
                ec_pop_key, ec_pop_idx = min(key_idx_to_rank.items(), key=lambda tup: tup[1])[0]

            if self.c_mon is not None:
                evicted_ots.add(ec_pop_key[0])
                evicted_ots.add(ec_pop_key[1])
                evicted_ots.discard(None)

            self.ec_buf.remove_item(ec_pop_key, ec_pop_idx)

        # Define unused unique event ID (between 0 and ec_buf_size-1) to identify which buffer items belong to same event
        # NOTE: must do this after potential removal if buffer is full s.t. new event ID is freed up in extreme case that there are ec_buf_size buffer items w/ ec_buf_size different unique event IDs
        if unique_event_id is None:
            avail_event_ids = set(range(self.ec_buf_size)) - self.ec_buf.get_buffered_nested_values_for_key(EVENT_ID)
            buf_dict[EVENT_ID] = next(iter(avail_event_ids))
        else:
            buf_dict[EVENT_ID] = unique_event_id
        if buf_dict[EVENT_ID] is None:
            raise RuntimeError('Cannot assign unique event ID to new TOTeM EC-buffer item!')

        # Insert/update buffer item for EC
        self.ec_buf.insert_new_item_by_cp(buf_key, buf_dict)

        if self.c_mon is not None:
            for hit_miss_ot in hit_miss_ots:
                self.c_mon.register_buf_insertion(num_hits, hit_miss_ot, self.ec_buf.buf_name)

            if len(evicted_ots) > 0:
                buffered_ots = set()
                for ec_buf_key in self.ec_buf.buf:
                    buffered_ots.add(ec_buf_key[0])
                    buffered_ots.add(ec_buf_key[1])
                buffered_ots.discard(None)

                for evicted_ot in evicted_ots:
                    full_eviction = evicted_ot not in buffered_ots
                    self.c_mon.register_buf_eviction(evicted_ot, full_eviction, self.ec_buf.buf_name)

        return buf_dict[EVENT_ID]
    
    def create_monitor_dataframes(self) -> None:
        """
        Turns information collected by runtime or cache monitor, if enabled, into DataFrames.

        Returns
        -------
        None
        """
        if self.rt_mon is not None:
            self.rt_mon.create_dataframes()
        elif self.c_mon is not None:
            self.c_mon.create_dataframes()
        else:
            return

    def __str__(self) -> str:
        """
        Creates string describing streaming TOTeM including its parameters (cache policy etc.) and items buffered in each model buffer in tabular format.

        Returns
        -------
        str
        """
        tr_buf_cols = [TR_BUF_KEY, OBJECT_TYPE, FIRST_SEEN, LAST_SEEN]
        ec_buf_cols = [EC_BUF_KEY, EVENT_ID, EVENT_CARD]
        lc_buf_cols = [LC_BUF_KEY, OBJECT_PAIR]

        if self.cp in [CachePolicy.LFU, CachePolicy.LFU_DA]:
            for buf_cols in [tr_buf_cols, ec_buf_cols, lc_buf_cols]:
                buf_cols.append(FREQUENCY)

        if self.cp == CachePolicy.LFU_DA:
            for buf_cols in [tr_buf_cols, ec_buf_cols, lc_buf_cols]:
                buf_cols.append(CACHE_AGE)
        
        ret = f'Coupled removal for buffered TOTeM model: {self.coupled_removal}\n'
        ret += self.tr_buf.to_string(tr_buf_cols) + self.ec_buf.to_string(ec_buf_cols) + self.lc_buf.to_string(lc_buf_cols)
        if self.pp_buf is not None:
            ret += self.pp_buf.to_string()

        return ret


class OcpnBuffer(object):
    """
    Represents a streaming Object-Centric Petri Net.
    The internal model buffers are maintained via a cache policy that is optionally combined with a priority policy.
    The streaming OCPN internally maintains a streaming OC-DFG (with shared model buffers or separate ones per object type) alongside an event-activity buffer.

    Attributes
    ----------
    node_buf_size : int
        Maximum number of buffered items in OC-DFG node buffer.
    arc_buf_size : int
        Maximum number of buffered items in OC-DFG arc buffer.
    ea_buf_size : int
        Maximum number of buffered items in event-activity buffer.
    cp : CachePolicy
        Cache policy used to replace model-buffer items when respective buffer is full.
    use_mixed_ocdfg_buf : bool
        If enabled, model buffers of internal streaming OC-DFG are shared by all object types.
    max_counter : int
        Maximum counter value (e.g. for key frequency or cache age) before reset.
    coupled_removal : bool
        If enabled, model buffers are synchronized on the basis of object types if model buffers are shared or on basis of object IDs if streaming OC-DFG has separate buffers per object type.
    pp_buf : PrioPolicyBuffer
        Optional priority-policy buffer that is maintained alongside event-activity buffer for removal decision based on additional object-centric characteristics.
    pp_buf_dfgs : PrioPolicyBuffer
        Optional priority-policy buffer that is maintained alongside streaming DFGs for removal decision based on additional object-centric characteristics.
    c_mon : CacheMonitor
        Optional CacheMonitor object for collecting information about cache behavior during stream processing; used for evaluation.
    rt_mon : RuntimeMonitor
        Optional RuntimeMonitor object for collecting information about runtime requirements during stream processing; used for evaluation.
    ocdfg_buf : Union[OcdfgBuffer, OcdfgBufferPerObjectType]
        Streaming OC-DFG.
    ea_buf : BufferOfDictLists
        Model buffer tracking information about event-activity pairs and number of involved objects per type.
    """
    def __init__(
            self, 
            node_buf_size : int, 
            arc_buf_size : int, 
            ea_buf_size : int, 
            cp : CachePolicy, 
            use_mixed_ocdfg_buf : bool = True, 
            max_counter : int = 10000, 
            coupled_removal : bool = False, 
            pp_buf : PrioPolicyBuffer = None, 
            pp_buf_dfgs : PrioPolicyBuffer = None, 
            cache_monitor : CacheMonitor = None, 
            runtime_monitor : RuntimeMonitor = None
        ) -> None:
        """
        Initializes an OcpnBuffer object.

        Parameters
        ----------
        node_buf_size : int
            Maximum number of buffered items in OC-DFG node buffer.
        arc_buf_size : int
            Maximum number of buffered items in OC-DFG arc buffer.
        ea_buf_size : int
            Maximum number of buffered items in event-activity buffer.
        cp : CachePolicy
            Cache policy used to replace model-buffer items when respective buffer is full.
        use_mixed_ocdfg_buf : bool, default=True
            If enabled, model buffers of internal streaming OC-DFG are shared by all object types.
        max_counter : int, default=10000
            Maximum counter value (e.g. for key frequency or cache age) before reset.
        coupled_removal : bool, default=False
            If enabled, model buffers are synchronized on the basis of object types if model buffers are shared or on basis of object IDs if streaming OC-DFG has separate buffers per object type.
        pp_buf : PrioPolicyBuffer, default=None
            Optional priority-policy buffer that is maintained alongside event-activity buffer for removal decision based on additional object-centric characteristics.
        pp_buf_dfgs : PrioPolicyBuffer, default=None
            Optional priority-policy buffer that is maintained alongside streaming DFGs for removal decision based on additional object-centric characteristics.
        cache_monitor : CacheMonitor, default=None
            Optional CacheMonitor object for collecting information about cache behavior during stream processing; used for evaluation.
        runtime_monitor : RuntimeMonitor, default=None
            Optional RuntimeMonitor object for collecting information about runtime requirements during stream processing; used for evaluation.
        
        Raises
        ------
        AttributeError
            Error occurs if shared vs. separate OC-DFG buffers are assigned an object- or type-based priority policy respectively. Error also occurs if event-activity buffer is assigned object-based priority policy or when running cache and runtime monitor simultaneously.
        """
        self.node_buf_size = node_buf_size
        self.arc_buf_size = arc_buf_size
        self.ea_buf_size = ea_buf_size      # buffer for tracking activity-OT-event pairs for deciding on double vs. single arcs in Petri net
        self.cp = cp
        self.use_mixed_ocdfg_buf = use_mixed_ocdfg_buf
        self.max_counter = max_counter
        self.coupled_removal = coupled_removal

        # Check for allowed combinations of (OC-)DFG buffers and priority-policy buffers per object/OT
        if not use_mixed_ocdfg_buf:
            if not isinstance(pp_buf_dfgs, (PPBStridePerObject, PPBLifespanPerObject, type(None))):
                raise AttributeError(f'Combining OT-based OCPN buffer for OT {self.ot} and priority-policy buffer based on {pp_buf.pp.value} not feasible. Need priority policy on object basis.')
            else:
                self.pp_buf_dfgs = pp_buf_dfgs
        if isinstance(pp_buf, (PPBStridePerObject, PPBLifespanPerObject)):
            raise AttributeError(f'Combining mixed OCPN buffers (OC-DFG buffers if enabled, event-activity buffer) and priority-policy buffer based on {pp_buf.pp.value} not feasible. Need priority policy on OT basis.')
        else:
            self.pp_buf = pp_buf
        
        # Set up monitors for evaluation
        if cache_monitor is not None and runtime_monitor is not None:
            raise AttributeError(f'Running CacheMonitor and RuntimeMonitor at the same time not recommended since CacheMonitor slows down computations! Pick one.')
        
        # Need to count stream items from "outside" individual OcdfgBuffers since their counts may start at different times
        self.stream_item_count = 0
        self.c_mon = cache_monitor
        if self.c_mon is not None and self.c_mon.total_model_buf_sizes is None and use_mixed_ocdfg_buf:
            self.c_mon.set_total_buf_sizes(node_buf_size + arc_buf_size + ea_buf_size)
        self.rt_mon = runtime_monitor

        # Initialize underyling (OC-)DFG model buffer and event-activity buffer specific to OCPN discovery
        if use_mixed_ocdfg_buf:
            self.ocdfg_buf = OcdfgBuffer(
                node_buf_size, 
                arc_buf_size, 
                cp, 
                max_counter=max_counter, 
                coupled_removal=coupled_removal, 
                pp_buf=None,                # need to update PPB externally due to additional event-activity buffer that also needs to be synchronized
                cache_monitor=self.c_mon, 
                runtime_monitor=self.rt_mon
            )
        else:
            self.ocdfg_buf = OcdfgBufferPerObjectType(
                node_buf_size, 
                arc_buf_size, 
                cp, 
                max_counter, 
                coupled_removal=coupled_removal, 
                pp_buf=pp_buf_dfgs, 
                cache_monitor=self.c_mon, 
                runtime_monitor=self.rt_mon
            )
        self.ea_buf = BufferOfDictLists(ea_buf_size, cp, 'OCPN event-activity buffer', EA_BUF_KEY, max_counter=max_counter)

    def process_stream(self, stream : list[Union[Event, O2OUpdate, ObjectAttributeUpdate]]) -> None:
        """
        Updates streaming representation by iterating over all stream items from start to finish.

        Parameters
        ----------
        stream : list[Union[Event, O2OUpdate, ObjectAttributeUpdate]]
            Object-centric event stream simulated as time-ordered list of stream items.

        Returns
        -------
        None
        """
        # Register stream size for monitoring
        if self.c_mon is not None:
            self.c_mon.register_stream_size(stream)
        
        # Record start time of stream processing for monitoring
        if self.rt_mon is not None:
            stream_start_time = time.time_ns()
        
        for stream_item in stream:
            # Increase stream item count globally for monitoring separate model buffers
            if self.c_mon is not None:
                self.c_mon.increase_stream_item_count()
                
                if self.c_mon.check_for_ot_frac_per_buffer():
                    if self.use_mixed_ocdfg_buf:
                        self.c_mon.register_ot_frac_for_mixed_buffer(self.ocdfg_buf.node_buf, nested_ot_key=OBJECT_TYPE)    
                        self.c_mon.register_ot_frac_for_mixed_buffer(self.ocdfg_buf.arc_buf, nested_ot_key=OBJECT_TYPE) 
                    else:
                        self.c_mon.register_ot_frac_for_single_ot_buffers({ot: dfg.node_buf for ot, dfg in self.ocdfg_buf.dfg_bufs.items()})
                        self.c_mon.register_ot_frac_for_single_ot_buffers({ot: dfg.arc_buf for ot, dfg in self.ocdfg_buf.dfg_bufs.items()})
                    self.c_mon.register_ot_frac_for_mixed_buffer(self.ea_buf, nested_ot_key=OBJECT_TYPE)
            elif self.rt_mon is not None:
                self.rt_mon.increase_stream_item_count()
            
            # Update OCPN buffers for incoming Events
            if isinstance(stream_item, Event):
                if self.rt_mon is not None:
                    item_processing_start = time.time_ns()
                
                # Create buffered DFG models for involved object types
                if not self.use_mixed_ocdfg_buf:
                    curr_ots = set([d['objectType'] for d in stream_item.e2o_relations])
                    for ot in curr_ots:
                        if ot not in self.ocdfg_buf.dfg_bufs:
                            self.ocdfg_buf.dfg_bufs[ot] = OcdfgBuffer(
                                self.node_buf_size, 
                                self.arc_buf_size, 
                                self.cp, 
                                ot=ot, 
                                max_counter=self.max_counter, 
                                coupled_removal=self.coupled_removal, 
                                pp_buf=deepcopy(self.pp_buf_dfgs), 
                                cache_monitor=self.c_mon, 
                                runtime_monitor=self.rt_mon
                            )
                
                self.update_buffers(stream_item)
                self.clean_up_buffers()

                # Monitor total item-processing time
                if self.rt_mon is not None:
                    self.rt_mon.register_item_processing_time(time.time_ns() - item_processing_start, stream_item.__class__.__name__)
        
        # Monitor total stream-processing time
        if self.rt_mon is not None:
            self.rt_mon.register_total_stream_processing_time(time.time_ns() - stream_start_time)
        
        # Record total buf sizes for DFG-based OC-DFG post-stream-processing based on how many OTs there are
        if self.c_mon is not None and not self.use_mixed_ocdfg_buf:
            self.c_mon.set_total_buf_sizes(self.ea_buf_size + (self.arc_buf_size + self.node_buf_size) * len(self.ocdfg_buf.dfg_bufs))
    
    def clean_up_buffers(self) -> None:
        """
        Synchronizes model buffers with each other in case of coupled removal, then synchronizes optional priority-policy buffer with model buffers.
        
        Returns
        -------
        None
        """
        if (self.coupled_removal or not isinstance(self.pp_buf, (type(None), PPBCustom))) and self.use_mixed_ocdfg_buf:
            # Monitor buffer-clean-up time
            if self.rt_mon is not None:
                clean_up_start_time = time.time_ns()

            # Perform buffer clean-up based on shared OTs between OC-DFG node buffer, arc buffer, and OCPN event-activity buffer
            node_buf_ots = self.ocdfg_buf.node_buf.get_buffered_nested_values_for_key(OBJECT_TYPE)
            arc_buf_ots = self.ocdfg_buf.arc_buf.get_buffered_nested_values_for_key(OBJECT_TYPE)
            ea_buf_ots = self.ea_buf.get_buffered_nested_values_for_key(OBJECT_TYPE)

            # NOTE: coupled removal (on OT basis) only relevant if underlying OCDFG buffer is mixed
            if self.coupled_removal:
                model_buf_ots = node_buf_ots.intersection(arc_buf_ots).intersection(ea_buf_ots)
                self.ocdfg_buf.node_buf.reduce_to_oids_ots(model_buf_ots, nested_key=OBJECT_TYPE)
                self.ocdfg_buf.arc_buf.reduce_to_oids_ots(model_buf_ots, nested_key=OBJECT_TYPE)
                self.ea_buf.reduce_to_oids_ots(model_buf_ots, nested_key=OBJECT_TYPE)
            
                if not isinstance(self.pp_buf, (type(None), PPBCustom)):
                    self.pp_buf.clean_up(model_buf_ots)
            else:
                if not isinstance(self.pp_buf, (type(None), PPBCustom)):
                    model_buf_ots = node_buf_ots.union(arc_buf_ots).union(ea_buf_ots)
                    self.pp_buf.clean_up(model_buf_ots)
            
            # Register buffer-clean-up time
            if self.rt_mon is not None:
                self.rt_mon.register_buf_update_time(time.time_ns() - clean_up_start_time, 'OCPN buffer clean-up')

            # Monitor size of PPB that's shared across entire buffered OCPN
            if self.c_mon is not None and not isinstance(self.pp_buf, type(None)):
                self.c_mon.update_ppb_size(len(self.pp_buf))

        elif not self.use_mixed_ocdfg_buf:
            # Monitor buffer-clean-up time
            if self.rt_mon is not None:
                clean_up_start_time = time.time_ns()
            
            # NOTE: event-activity and priority-policy buffer don't need to be updated since no OTs are evicted w/ individual DFGs 
            # Clean up individual DFG buffers based on OID
            if self.coupled_removal:
                for dfg in self.ocdfg_buf.dfg_bufs.values():
                    dfg.clean_up_buffers()
            
            # Register buffer-clean-up time
            if self.rt_mon is not None:
                self.rt_mon.register_buf_update_time(time.time_ns() - clean_up_start_time, 'OCPN buffer clean-up')

            # Monitor sum of PPB sizes across each DFG and event-activity buffer
            if self.c_mon is not None:
                new_ppb_size = 0
                if not isinstance(self.pp_buf, type(None)):
                    new_ppb_size += len(self.pp_buf)
                if not isinstance(self.pp_buf_dfgs, type(None)):
                    for dfg in self.ocdfg_buf.dfg_bufs.values():
                        new_ppb_size += len(dfg.pp_buf)
                self.c_mon.update_ppb_size(new_ppb_size)
        
        else:
            return
            
    def update_buffers(self, stream_item : Event) -> None:
        """
        Update streaming OCPN for in-coming event.

        Parameters
        ----------
        stream_item : Event
            Event from stream for which streaming OCPN is updated.

        Returns
        -------
        None
        """
        if isinstance(stream_item, Event):
            ot_to_pp_rank = None
            if self.pp_buf is not None:
                if not isinstance(self.pp_buf, (PPBCustom)):
                    if self.rt_mon is not None:
                        ppb_start_time = time.time_ns()

                    self.pp_buf.update(stream_item)

                    if self.rt_mon is not None:
                        self.rt_mon.register_buf_update_time(time.time_ns() - ppb_start_time, self.pp_buf.buf_name)
                
                ot_to_pp_rank = self.pp_buf.get_normalized_rank_by_pp()

                if self.use_mixed_ocdfg_buf:
                    self.ocdfg_buf.update_buffers(stream_item, ot_to_pp_rank)
                else:
                    # Update buffered DFG models for invovled object types
                    for dfg_buf in self.ocdfg_buf.dfg_bufs.values(): 
                        dfg_buf.update_buffers(stream_item)
            
            # Update event-activity buffer
            act = stream_item.activity
            curr_ots_to_oids = dict()
            for e2o_dict in stream_item.e2o_relations:
                ot = e2o_dict['objectType']
                oid = e2o_dict['objectId']
                curr_ots_to_oids.setdefault(ot, set())
                curr_ots_to_oids[ot].add(oid)

            unique_event_id = None
            for curr_ot, oid_set in curr_ots_to_oids.items():
                has_single_obj = len(oid_set) == 1
                if self.rt_mon is not None:
                    ea_start_time = time.time_ns()

                unique_event_id = self.update_ea_buf(act, {OBJECT_TYPE: curr_ot, HAS_SINGLE_OBJ: has_single_obj}, unique_event_id, ot_to_pp_rank)

                if self.rt_mon is not None:
                    self.rt_mon.register_buf_update_time(time.time_ns() - ea_start_time, self.ea_buf.buf_name)
        else:
            return

    def update_ea_buf(self, buf_key : str, buf_dict : dict[str, Any], unique_event_id : int = None, ot_to_pp_rank : dict[str, float] = None) -> int:
        """
        Insert new item into OCPN event-activity buffer.

        Parameters
        ----------
        buf_key : str
            Key of new buffer item, i.e. activity.
        buf_dict : dict[str, Any]
            Key-value pairs of associated value dictionary that is inserted for given activity.
        unique_event_id : int, default=None
            Event ID that is unique w.r.t. IDs already buffered that uniquely identifies event associated with new buffer item.
        ot_to_pp_rank : dict[str, float], default=None
            Optional mapping of object types to priority-policy-based rank.
        
        Returns
        -------
        int
            Event ID that uniquely identifies event associated with new buffer item among already buffered events.
        """
        if self.c_mon is not None:
            num_hits = 1 if buf_key in self.ea_buf.buf else 0
            hit_miss_ot = buf_dict[OBJECT_TYPE]
            evicted_ot = None
        
        # Remove activities according to cache policy
        if self.ea_buf.is_full():
            if ot_to_pp_rank is None:
                act_pop_key, act_pop_idx = self.ea_buf.get_removal_key_by_cp()
            else:
                key_idx_to_rank = dict()
                key_idx_to_cp_rank = self.ea_buf.get_normalized_rank_by_cp()

                for act_key, list_id in key_idx_to_cp_rank:
                    ot_key = self.ea_buf.buf[act_key][list_id][OBJECT_TYPE]
                    if ot_key in ot_to_pp_rank:
                        key_idx_to_rank[(act_key, list_id)] = key_idx_to_cp_rank[(act_key, list_id)] + ot_to_pp_rank[ot_key]
                    else:
                        key_idx_to_rank[(act_key, list_id)] = 2 * key_idx_to_cp_rank[(act_key, list_id)]

                act_pop_key, act_pop_idx = min(key_idx_to_rank.items(), key=lambda tup: tup[1])[0]
            
            if self.c_mon is not None:
                evicted_ot = self.ea_buf.buf[act_pop_key][act_pop_idx][OBJECT_TYPE]
            
            self.ea_buf.remove_item(act_pop_key, act_pop_idx)
        
        # Define unused unique event ID (between 0 and ea_buf_size-1) to identify which buffer items belong to the same event
        if unique_event_id is None:
            avail_event_ids = set(range(self.ea_buf_size)) - self.ea_buf.get_buffered_nested_values_for_key(EVENT_ID)
            buf_dict[EVENT_ID] = next(iter(avail_event_ids))
        else:
            buf_dict[EVENT_ID] = unique_event_id
        if buf_dict[EVENT_ID] is None:
            raise RuntimeError('Cannot assign unique event ID to new OCPN EA-buffer item!')

        # Insert/update buffer item for activity
        self.ea_buf.insert_new_item_by_cp(buf_key, buf_dict)

        if self.c_mon is not None:
            self.c_mon.register_buf_insertion(num_hits, hit_miss_ot, self.ea_buf.buf_name)

            if evicted_ot is not None:
                full_eviction = evicted_ot not in self.ea_buf.get_buffered_nested_values_for_key(OBJECT_TYPE)
                self.c_mon.register_buf_eviction(evicted_ot, full_eviction, self.ea_buf.buf_name)
        
        return buf_dict[EVENT_ID]
    
    def create_monitor_dataframes(self) -> None:
        """
        Turns information collected by runtime or cache monitor, if enabled, into DataFrames.

        Returns
        -------
        None
        """
        if self.rt_mon is not None:
            self.rt_mon.create_dataframes()
        elif self.c_mon is not None:
            self.c_mon.create_dataframes()
        else:
            return
    
    def __str__(self) -> str:
        """
        Creates string describing streaming OCPN including its parameters (cache policy etc.) and items buffered in each model buffer in tabular format.

        Returns
        -------
        str
        """
        ret = f'Coupled removal of OCPN model: {self.coupled_removal}\n'
        ret += f'Underlying (OC-)DFG buffer for OCPN model: {self.ocdfg_buf.__class__.__name__}\n'

        ea_buf_cols = [EA_BUF_KEY, OBJECT_TYPE, HAS_SINGLE_OBJ, EVENT_ID]
        if self.cp in [CachePolicy.LFU, CachePolicy.LFU_DA]:
            ea_buf_cols.append(FREQUENCY)
        if self.cp == CachePolicy.LFU_DA:
            ea_buf_cols.append(CACHE_AGE)
        ret += self.ea_buf.to_string(ea_buf_cols)
        
        if self.pp_buf is not None:
            ret += self.pp_buf.to_string()
        
        ret += str(self.ocdfg_buf)

        return ret