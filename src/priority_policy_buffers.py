"""\
Classes representing different types of priority policies maintained as buffers.
__author__: "Nina Löseke"
"""

from abc import ABC, abstractmethod
from typing import Any, Union
from enum import Enum
import pandas as pd
import numpy as np
from tabulate import tabulate
from vars import *
from utils import get_mean_timedelta, min_max_normalize_dict, Event, O2OUpdate


class PriorityPolicy(Enum):
    """
    Represents object-centric priority policies optionally used by model buffers.
    """

    STRIDE_OBJ = "stride per object"
    STRIDE_OT = "stride per OT"
    LIFESPAN_OBJ = "lifespan per object"
    LIFESPAN_OT = "lifespan per OT"
    OBJ_PER_EVENT = "#objects per event"
    OBJ_PER_OT = "#objects per OT"
    EVENTS_PER_OT = "#events per OT"
    CUSTOM_OT_ORDER = "custom OT order"


class PrioPolicyOrder(Enum):
    """
    Represents direction of priority policy.
    """

    MIN = 'min'
    MAX = 'max'


class PrioPolicyBuffer(ABC):
    """
    Abstract base class defining required functionality of a priority-policy buffer.
    """

    prio_order : PrioPolicyOrder
    buf : dict
    pp : PriorityPolicy

    def clean_up(self, rel_keys : set[str]) -> None:
        keys_pop = set(self.buf.keys()) - rel_keys
        for key_pop in keys_pop:
            self.buf.pop(key_pop)

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def get_normalized_rank_by_pp(self):
        # NOTE: make sure to add "+1" to values in key-value pairs since ranking might have to be inverted (if PrioPolicyOrder.MAX) via 1/val to avoid division by 0
        pass

    @abstractmethod
    def to_string(self):
        pass


class PPBStridePerObject(PrioPolicyBuffer):
    """
    Represents a priority-policy buffer that tracks stride per object.

    Attributes
    ----------
    window_size : int
        Maximum number of strides that are tracked per object in sliding-window fashion.
    prio_order : PrioPolicyOrder
        Direction of priority policy, e.g. for MIN, object with smallest average stride is most likely to be removed if the model buffer is full.
    pp : PriorityPolicy
        Enum of corresponding priority policy.
    buf_name : str
        Name and direction of priority policy.
    buf : dict[str, dict[str, Any]]
        Dictionary containing priority-policy buffer items, i.e. mapping objects to last-seen timestamps and latest strides.
    """
    
    def __init__(self, prio_order : PrioPolicyOrder, window_size : int = 10):
        """
        Initializes a PPBStridePerObject object.

        Parameters
        ----------
        prio_order : PrioPolicyOrder
            Direction of priority policy.
        window_size : int, default=10
            Maximum number of strides that are tracked per object in sliding-window fashion.
        """
        self.window_size = window_size
        self.prio_order = prio_order

        self.pp = PriorityPolicy.STRIDE_OBJ
        self.buf_name = f'{prio_order.value} {self.pp.value}'
        # OID -> {LAST_SEEN: timestamp, STRIDES: [timedelta, ...]}
        self.buf : dict[str, dict[str, Any]] = dict()

    def __len__(self) -> int:
        return len(self.buf)

    def update(self, stream_item : Union[Event, O2OUpdate]) -> None:
        """
        Updates strides for currently buffered objects or adds new objects.

        Parameters
        ----------
        stream_item : Union[Event, O2OUpdate]
            In-coming event or object-to-object-update whose associated objects update the buffer.

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
            If unsupported stream item is given.
        """
        curr_ts = stream_item.time
        new_keys = list()
        if isinstance(stream_item, Event):
            new_keys = [d['objectId'] for d in stream_item.e2o_relations]
        elif isinstance(stream_item, O2OUpdate):
            new_keys = [stream_item.id, stream_item.target_id]
        else:
            raise NotImplementedError(f'Stride-based PriorityPolicy buffer not implemented for given stream item of type {type(stream_item)}.')
        
        for new_key in new_keys:
            if new_key not in self.buf:
                self.buf[new_key] = {LAST_SEEN: curr_ts, STRIDES: list()}
            else:
                key_dict = self.buf[new_key]
                if len(key_dict[STRIDES]) >= self.window_size:
                    key_dict[STRIDES].pop(0)
                key_dict[STRIDES].append(curr_ts - key_dict[LAST_SEEN])
                key_dict[LAST_SEEN] = curr_ts
    
    def get_normalized_rank_by_pp(self) -> dict[str, float]:
        """
        Depending on direction of priority policy, min-max-normalized "rank" for each buffered object is determined based on its average stride. The object with the smallest rank is most likely to be removed.

        Returns
        -------
        dict[str, float]
            Mapping of objects to min-max-normalized ranks in [0, 1].
        """
        # Use maximum last seen timestamp to derive minimum possible stride for objects that were only seen once thus far and therefore have an empty STRIDES list
        max_last_seen = max([buf_dict[LAST_SEEN] for buf_dict in self.buf.values()])

        if self.prio_order == PrioPolicyOrder.MIN:
            key_to_val = {key: get_mean_timedelta(val[STRIDES]).total_seconds() if len(val[STRIDES]) > 0 else (max_last_seen - val[LAST_SEEN]).total_seconds() for key, val in self.buf.items()}
        else:
            # Give lowest ranking to object w/ largest average stride, therefore more likely to be removed from buffer
            key_to_val = {key: 1/(get_mean_timedelta(val[STRIDES]).total_seconds()+1) if len(val[STRIDES]) > 0 
                          else 1/((max_last_seen - val[LAST_SEEN]).total_seconds()+1) for key, val in self.buf.items()}

        return min_max_normalize_dict(key_to_val)
    
    def to_string(self) -> str:
        """
        Creates string describing priority-policy buffer including its parameters (e.g. direction) and buffered objects in tabular format.

        Returns
        -------
        str
            Output string describing parameters and content of buffer.
        """
        ret_str = f'Priority-policy buffer characteristics:\n - priority policy: {self.pp.value}\n - most likely to get removed for {self.prio_order.value} value\n - buffer size: {len(self)}\n - window size: {self.window_size}\n'

        # Unroll buffer structure: OID -> {LAST_SEEN: timestamp, STRIDES: [timedelta, ...]}
        num_strides_key = '# strides'
        avg_stride_key = 'avg stride'
        unrolled_buf = {OID: list(), LAST_SEEN: list(), num_strides_key: list(), avg_stride_key: list()}

        for oid, stride_dict in self.buf.items():
            unrolled_buf[OID].append(oid)
            unrolled_buf[LAST_SEEN].append(stride_dict[LAST_SEEN])
            unrolled_buf[num_strides_key].append(len(stride_dict[STRIDES]))
            unrolled_buf[avg_stride_key].append(get_mean_timedelta(stride_dict[STRIDES]) if len(stride_dict[STRIDES]) > 0 else '--')

        buf_df = pd.DataFrame.from_dict(unrolled_buf)
        ret_str += tabulate(buf_df, headers='keys', tablefmt='psql', showindex=True)
        ret_str += '\n'

        return ret_str
    

class PPBStridePerObjectType(PrioPolicyBuffer):
    """
    Represents a priority-policy buffer that tracks stride per object type.

    Attributes
    ----------
    window_size : int
        Maximum number of strides that are tracked per object in sliding-window fashion.
    max_obj_per_ot : int
        Maximum number of objects to track per object type.
    prio_order : PrioPolicyOrder
        Direction of priority policy.
    freeze_objects : bool
        If true, objects per object type that are buffered first are "frozen" and tracked throughout entire stream processing. If False, the tracked objects are replaced with more more recent, unseen ones.
    max_obj_idle : pd.Timedelta
        Maximum about of idle time after which a buffered object is evicted to make space for new ones.
    pp : PriorityPolicy
        Enum of corresponding priority policy.
    buf_name : str
        Name and direction of priority policy.
    buf : dict[str, dict[str, dict[str, Any]]]
        Dictionary containing priority-policy buffer items, i.e. mapping object types to objects and their last-seen timestamps and latest strides.
    """

    def __init__(self, prio_order : PrioPolicyOrder, window_size : int = 10, max_obj_per_ot : int = 5, freeze_or_max_idle : Union[bool, pd.Timedelta] = None):
        """
        Initializes a PPBStridePerObjectType object.

        Parameters
        ----------
        prio_order : PrioPolicyOrder
            Direction of priority policy.
        window_size : int, default=10
            Maximum number of strides that are tracked per object in sliding-window fashion.
        max_obj_per_ot : int, default=5
            Maximum number of objects to track per object type.
        freeze_or_max_idle : Union[bool, pd.Timedelta], default=None
            Indicates if initially tracked objects are "frozen", or indicates maximum idle time after which buffered objects are replaced.
        """
        self.window_size = window_size
        self.max_obj_per_ot = max_obj_per_ot
        self.prio_order = prio_order
        self.freeze_objects = False
        self.max_obj_idle = None
        if isinstance(freeze_or_max_idle, bool):
            self.freeze_objects = freeze_or_max_idle
        elif isinstance(freeze_or_max_idle, pd.Timedelta):
            self.max_obj_idle = freeze_or_max_idle
        
        self.pp = PriorityPolicy.STRIDE_OT
        self.buf_name = f'{prio_order.value} {self.pp.value}'
        # OT -> {OID: {LAST_SEEN: timestamp, STRIDES: [timedelta, ...]}, ...}
        self.buf : dict[str, dict[str, dict[str, Any]]] = dict()

    def __len__(self) -> int:
        buf_len = 0
        for ot in self.buf:
            buf_len += len(self.buf[ot])
        return buf_len

    def update(self, stream_item : Union[Event, O2OUpdate]) -> None:
        """
        Updates strides for currently buffered objects or adds new objects.

        Parameters
        ----------
        stream_item : Union[Event, O2OUpdate]
            In-coming event or object-to-object-update whose associated objects update the buffer.

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
            If unsupported stream item is given.
        """
        curr_ts = stream_item.time
        new_keys = list()
        if isinstance(stream_item, Event):
            new_keys = [(d['objectType'], d['objectId']) for d in stream_item.e2o_relations]
        elif isinstance(stream_item, O2OUpdate):
            new_keys = [(stream_item.type, stream_item.id), (stream_item.target_type, stream_item.target_id)]
        else:
            raise NotImplementedError(f'Stride-based PriorityPolicy buffer not implemented for given stream item of type {type(stream_item)}.')
        
        for new_ot_key, new_oid_key in new_keys:
            if new_ot_key not in self.buf:
                self.buf[new_ot_key] = {new_oid_key: {LAST_SEEN: curr_ts, STRIDES: list()}}
            else:
                ot_key_dict = self.buf[new_ot_key]
                # If maximum #objects per OT in buffer reached, move window if not "freezing objects" or waiting until maximum idle time per object has passed
                if not self.freeze_objects and self.max_obj_idle is None and len(ot_key_dict) >= self.max_obj_per_ot:
                    ot_key_dict.pop(next(iter(ot_key_dict)))
                elif not self.freeze_objects and self.max_obj_idle is not None and len(ot_key_dict) >= self.max_obj_per_ot:
                    for oid_key, oid_dict in list(ot_key_dict.items()):
                        if curr_ts - oid_dict[LAST_SEEN] >= self.max_obj_idle:
                            ot_key_dict.pop(oid_key)
                
                if new_oid_key in ot_key_dict:
                    oid_key_dict = ot_key_dict[new_oid_key]
                    # If maximum #strides per object in buffer reached, move window
                    if len(oid_key_dict[STRIDES]) == self.window_size:
                        oid_key_dict[STRIDES].pop(0)

                    oid_key_dict[STRIDES].append(curr_ts - oid_key_dict[LAST_SEEN])
                    oid_key_dict[LAST_SEEN] = curr_ts
                else:
                    if len(ot_key_dict) < self.max_obj_per_ot:
                        ot_key_dict[new_oid_key] = {LAST_SEEN: curr_ts, STRIDES: list()}
    
    def get_normalized_rank_by_pp(self) -> dict[str, float]:
        """
        Depending on direction of priority policy, min-max-normalized "rank" for each buffered object type is determined based on the average stride of its buffered objects. The type with the smallest rank is most likely to be removed.

        Returns
        -------
        dict[str, float]
            Mapping of object types to min-max-normalized ranks in [0, 1].
        """
        # Use maximum last seen timestamp to derive minimum possible stride for objects that were only seen once thus far and therefore have an empty STRIDES list
        ot_to_max_last_seen = dict()
        for ot in self.buf:
            ot_to_max_last_seen[ot] = max([self.buf[ot][oid][LAST_SEEN] for oid in self.buf[ot]])
        
        key_to_val = dict()
        for ot in self.buf:
            ot_strides = list()
            for oid in self.buf[ot]:
                oid_strides = self.buf[ot][oid][STRIDES]
                if len(oid_strides) > 0:
                    ot_strides.extend(oid_strides)
                else:
                    ot_strides.append(ot_to_max_last_seen[ot] - self.buf[ot][oid][LAST_SEEN])
            
            if self.prio_order == PrioPolicyOrder.MIN:
                key_to_val[ot] = get_mean_timedelta(ot_strides).total_seconds()
            else:
                key_to_val[ot] = 1/(get_mean_timedelta(ot_strides).total_seconds()+1)

        return min_max_normalize_dict(key_to_val)
    
    def to_string(self) -> str:
        """
        Creates string describing priority-policy buffer including its parameters (e.g. direction) and buffered object types in tabular format.

        Returns
        -------
        str
            Output string describing parameters and content of buffer.
        """
        ret_str = f'Priority-policy buffer characteristics:\n - priority policy: {self.pp.value}\n - most likely to get removed for {self.prio_order.value} value\n - buffer size: {len(self)}\n - window size: {self.window_size}\n - max # objects per OT: {self.max_obj_per_ot}\n'

        # Unroll buffer structure: OT -> {OID: {LAST_SEEN: timestamp, STRIDES: [timedelta, ...]}, ...}
        num_strides_key = '# strides'
        avg_stride_key = 'avg stride'
        unrolled_buf = {OBJECT_TYPE: list(), OID: list(), LAST_SEEN: list(), num_strides_key: list(), avg_stride_key: list()}

        for ot, oid_dict in self.buf.items():
            for oid, stride_dict in oid_dict.items():
                unrolled_buf[OBJECT_TYPE].append(ot)
                unrolled_buf[OID].append(oid)
                unrolled_buf[LAST_SEEN].append(stride_dict[LAST_SEEN])
                unrolled_buf[num_strides_key].append(len(stride_dict[STRIDES]))
                unrolled_buf[avg_stride_key].append(get_mean_timedelta(stride_dict[STRIDES]) if len(stride_dict[STRIDES]) > 0 else '--')

        buf_df = pd.DataFrame.from_dict(unrolled_buf)
        ret_str += tabulate(buf_df, headers='keys', tablefmt='psql', showindex=True)
        ret_str += '\n'

        return ret_str


class PPBLifespanPerObject(PrioPolicyBuffer):
    """
    Represents a priority-policy buffer that tracks lifespan per object.

    Attributes
    ----------
    prio_order : PrioPolicyOrder
        Direction of priority policy.
    pp : PriorityPolicy
        Enum of corresponding priority policy.
    buf_name : str
        Name and direction of priority policy.
    buf : dict[str, dict[str, pd.Timestamp]]
        Dictionary containing priority-policy buffer items, i.e. mapping objects to last-seen and first-seen timestamps.
    """

    def __init__(self, prio_order : PrioPolicyOrder):
        """
        Initializes a PPBLifespanPerObject object.

        Parameters
        ----------
        prio_order : PrioPolicyOrder
            Direction of priority policy.
        """
        self.prio_order = prio_order
        
        self.pp = PriorityPolicy.LIFESPAN_OBJ
        self.buf_name = f'{prio_order.value} {self.pp.value}'
        # OID -> {LAST_SEEN: timestamp, FIRST_SEEN: timestamp}
        self.buf : dict[str, dict[str, pd.Timestamp]] = dict()

    def __len__(self) -> int:
        return len(self.buf)

    def update(self, stream_item : Union[Event, O2OUpdate]) -> None:
        """
        Updates lifespans for currently buffered objects or adds new objects.

        Parameters
        ----------
        stream_item : Union[Event, O2OUpdate]
            In-coming event or object-to-object-update whose associated objects update the buffer.

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
            If unsupported stream item is given.
        """
        curr_ts = stream_item.time
        new_keys = list()
        if isinstance(stream_item, Event):
            new_keys = [d['objectId'] for d in stream_item.e2o_relations]
        elif isinstance(stream_item, O2OUpdate):
            new_keys = [stream_item.id, stream_item.target_id]
        else:
            raise NotImplementedError(f'Lifespan-based PriorityPolicy buffer not implemented for given stream item of type {type(stream_item)}.')
        
        for new_key in new_keys:
            if new_key not in self.buf:
                self.buf[new_key] = {FIRST_SEEN: curr_ts, LAST_SEEN: curr_ts}
            else:
                self.buf[new_key][LAST_SEEN] = curr_ts

    def get_normalized_rank_by_pp(self) -> dict[str, float]:
        """
        Depending on direction of priority policy, min-max-normalized "rank" for each buffered object is determined based on its lifespan. The object with smallest rank is most likely to be removed.

        Returns
        -------
        dict[str, float]
            Mapping of objects to min-max-normalized ranks in [0, 1].
        """
        if self.prio_order == PrioPolicyOrder.MIN:
            key_to_val = {key: (val[LAST_SEEN] - val[FIRST_SEEN]).total_seconds() for key, val in self.buf.items()}
        else:
            # Give lowest ranking to object w/ largest average stride, therefore more likely to be removed from buffer
            key_to_val = {key: 1/((val[LAST_SEEN] - val[FIRST_SEEN]).total_seconds()+1) for key, val in self.buf.items()}

        return min_max_normalize_dict(key_to_val)
    
    def to_string(self) -> str:
        ret_str = f'Priority-policy buffer characteristics:\n - priority policy: {self.pp.value}\n - most likely to get removed for {self.prio_order.value} value\n - buffer size: {len(self)}\n'

        # Unroll buffer structure: OID -> {LAST_SEEN: timestamp, FIRST_SEEN: timestamp}
        lifespan_key = 'lifespan'
        unrolled_buf = {OID: list(), FIRST_SEEN: list(), LAST_SEEN: list(), lifespan_key: list()}

        for oid, oid_dict in self.buf.items():
            unrolled_buf[OID].append(oid)
            unrolled_buf[FIRST_SEEN].append(oid_dict[FIRST_SEEN])
            unrolled_buf[LAST_SEEN].append(oid_dict[LAST_SEEN])
            unrolled_buf[lifespan_key].append(oid_dict[LAST_SEEN] - oid_dict[FIRST_SEEN])

        buf_df = pd.DataFrame.from_dict(unrolled_buf)
        ret_str += tabulate(buf_df, headers='keys', tablefmt='psql', showindex=True)
        ret_str += '\n'

        return ret_str


class PPBLifespanPerObjectType(PrioPolicyBuffer):
    """
    Represents a priority-policy buffer that tracks lifespan per object type as average lifespan across associated objects.

    Attributes
    ----------
    prio_order : PrioPolicyOrder
        Direction of priority policy.
    max_obj_per_ot : int
        Maximum number of objects to track per object type.
    freeze_objects : bool
        If true, objects per object type that are buffered first are "frozen" and tracked throughout entire stream processing. If False, the tracked objects are replaced with more more recent, unseen ones.
    max_obj_idle : pd.Timedelta
        Maximum about of idle time after which a buffered object is evicted to make space for new ones.
    pp : PriorityPolicy
        Enum of corresponding priority policy.
    buf_name : str
        Name and direction of priority policy.
    buf : dict[str, dict[str, dict[str, pd.Timestamp]]]
        Dictionary containing priority-policy buffer items, i.e. mapping object types to objects and their last-seen and first-seen timestamps.
    """

    def __init__(self, prio_order : PrioPolicyOrder, max_obj_per_ot : int = 5, freeze_or_max_idle : Union[bool, pd.Timedelta] = None):
        """
        Initializes a PPBLifespanPerObjectType object.

        Parameters
        ----------
        prio_order : PrioPolicyOrder
            Direction of priority policy.
        max_obj_per_ot : int, default=5
            Maximum number of objects to track per object type.
        freeze_or_max_idle : Union[bool, pd.Timedelta], default=None
            Indicates if initially tracked objects are "frozen", or indicates maximum idle time after which buffered objects are replaced.
        """
        self.prio_order = prio_order
        self.max_obj_per_ot = max_obj_per_ot
        self.freeze_objects = False
        self.max_obj_idle = None
        if isinstance(freeze_or_max_idle, bool):
            self.freeze_objects = freeze_or_max_idle
        elif isinstance(freeze_or_max_idle, pd.Timedelta):
            self.max_obj_idle = freeze_or_max_idle
        
        self.pp = PriorityPolicy.LIFESPAN_OT
        self.buf_name = f'{prio_order.value} {self.pp.value}'
        # OT -> {OID: {FIRST_SEEN: timestamp, LAST_SEEN: timestamp}, ...}
        self.buf : dict[str, dict[str, dict[str, pd.Timestamp]]] = dict()

    def __len__(self) -> int:
        buf_len = 0
        for ot in self.buf:
            buf_len += len(self.buf[ot])
        return buf_len

    def update(self, stream_item : Union[Event, O2OUpdate]) -> None:
        """
        Updates lifespans for currently buffered objects or adds new objects.

        Parameters
        ----------
        stream_item : Union[Event, O2OUpdate]
            In-coming event or object-to-object-update whose associated objects update the buffer.

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
            If unsupported stream item is given.
        """
        curr_ts = stream_item.time
        new_keys = list()
        if isinstance(stream_item, Event):
            new_keys = [(d['objectType'], d['objectId']) for d in stream_item.e2o_relations]
        elif isinstance(stream_item, O2OUpdate):
            new_keys = [(stream_item.type, stream_item.id), (stream_item.target_type, stream_item.target_id)]
        else:
            raise NotImplementedError(f'Lifespan-based PriorityPolicy buffer not implemented for given stream item of type {type(stream_item)}.')
        
        for new_ot_key, new_oid_key in new_keys:
            if new_ot_key not in self.buf:
                self.buf[new_ot_key] = {new_oid_key: {FIRST_SEEN: curr_ts, LAST_SEEN: curr_ts}}
            else:
                ot_key_dict = self.buf[new_ot_key]
                # If max #objects per OT in buffer, move window if not "freezing objects" or waiting for maximum idle time per object to pass before removal
                if not self.freeze_objects and self.max_obj_idle is None and len(ot_key_dict) >= self.max_obj_per_ot:
                    ot_key_dict.pop(next(iter(ot_key_dict)))
                elif not self.freeze_objects and self.max_obj_idle is not None and len(ot_key_dict) >= self.max_obj_per_ot:
                    # Check if any current object for OT has reached max lifetime / period of inactivity to be marked as "dead"
                    for oid_key, oid_dict in list(ot_key_dict.items()):
                        if curr_ts - oid_dict[LAST_SEEN] >= self.max_obj_idle:
                            ot_key_dict.pop(oid_key)

                if new_oid_key in ot_key_dict:
                    ot_key_dict[new_oid_key][LAST_SEEN] = curr_ts
                else:
                    if len(ot_key_dict) < self.max_obj_per_ot:
                        ot_key_dict[new_oid_key] = {FIRST_SEEN: curr_ts, LAST_SEEN: curr_ts}
    
    def get_normalized_rank_by_pp(self) -> dict[str, float]:
        """
        Depending on direction of priority policy, min-max-normalized "rank" for each buffered object type is determined based on the average lifespan of its buffered objects. The type with the smallest rank is most likely to be removed.

        Returns
        -------
        dict[str, float]
            Mapping of object types to min-max-normalized ranks in [0, 1].
        """
        key_to_val = dict()
        for ot in self.buf:
            ot_lifespans = list()
            for oid in self.buf[ot]:
                oid_dict = self.buf[ot][oid]
                ot_lifespans.append(oid_dict[LAST_SEEN] - oid_dict[FIRST_SEEN])
            if self.prio_order == PrioPolicyOrder.MIN:
                key_to_val[ot] = get_mean_timedelta(ot_lifespans).total_seconds()
            else:
                key_to_val[ot] = 1/(get_mean_timedelta(ot_lifespans).total_seconds()+1)

        return min_max_normalize_dict(key_to_val)
    
    def to_string(self) -> str:
        """
        Creates string describing priority-policy buffer including its parameters (e.g. direction) and buffered object types in tabular format.

        Returns
        -------
        str
            Output string describing parameters and content of buffer.
        """
        ret_str = f'Priority-policy buffer characteristics:\n - priority policy: {self.pp.value}\n - most likely to get removed for {self.prio_order.value} value\n - buffer size: {len(self)}\n - max # objects per OT: {self.max_obj_per_ot}\n'

        # Unroll buffer structure: OT -> {OID: {FIRST_SEEN: timestamp, LAST_SEEN: timestamp}, ...}
        lifespan_key = 'lifespan'
        unrolled_buf = {OBJECT_TYPE: list(), OID: list(), FIRST_SEEN: list(), LAST_SEEN: list(), lifespan_key: list()}

        for ot, oid_dict in self.buf.items():
            for oid, lifespan_dict in oid_dict.items():
                unrolled_buf[OBJECT_TYPE].append(ot)
                unrolled_buf[OID].append(oid)
                unrolled_buf[FIRST_SEEN].append(lifespan_dict[FIRST_SEEN])
                unrolled_buf[LAST_SEEN].append(lifespan_dict[LAST_SEEN])
                unrolled_buf[lifespan_key].append(lifespan_dict[LAST_SEEN] - lifespan_dict[FIRST_SEEN])

        buf_df = pd.DataFrame.from_dict(unrolled_buf)
        ret_str += tabulate(buf_df, headers='keys', tablefmt='psql', showindex=True)
        ret_str += '\n'

        return ret_str


class PPBObjectsPerEvent(PrioPolicyBuffer):
    """
    Represents a priority-policy buffer that counts the average number of objects per event.

    Attributes
    ----------
    window_size : int
        Maximum number of events that are tracked per object type.
    prio_order : PrioPolicyOrder
        Direction of priority policy.
    pp : PriorityPolicy
        Enum of corresponding priority policy.
    buf_name : str
        Name and direction of priority policy.
    buf : dict[str, list[int]]
        Dictionary containing priority-policy buffer items, i.e. mapping object types to number of respective objects over latest events.
    """

    def __init__(self, prio_order : PrioPolicyOrder, window_size : int = 10):
        """
        Initializes a PPBObjectsPerEvent object.

        Parameters
        ----------
         prio_order : PrioPolicyOrder
            Direction of priority policy.
        window_size : int, default=10
            Maximum number of events per object type for which number of objects are tracked in sliding-window fashion.
        """
        self.window_size = window_size
        self.prio_order = prio_order
        
        self.pp = PriorityPolicy.OBJ_PER_EVENT
        self.buf_name = f'{prio_order.value} {self.pp.value}'
        # OT -> [#objects per event, ...]
        self.buf : dict[str, list[int]] = dict()

    def __len__(self) -> int:
        return len(self.buf)

    def update(self, stream_item : Event) -> None:
        """
        Updates number of objects per event for currently buffered object types or adds new types.

        Parameters
        ----------
        stream_item : Event
            In-coming event whose involved object types update the buffer.

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
            If unsupported stream item is given.
        """
        # Update #objects per event for currently buffered OTs or add new OTs to buffer
        count_per_ot = dict()
        if isinstance(stream_item, Event):
            for d in stream_item.e2o_relations:
                ot = d['objectType']
                count_per_ot[ot] = count_per_ot.get(ot, 0) + 1
        else:
            raise NotImplementedError(f'#Objects-per-event-based PriorityPolicy buffer not implemented for given stream item of type {type(stream_item)}.')
        
        for new_ot, new_count in count_per_ot.items():
            if new_ot not in self.buf:
                self.buf[new_ot] = [new_count]
            else:
                if len(self.buf[new_ot]) == self.window_size:
                    self.buf[new_ot].pop(0)
                self.buf[new_ot].append(new_count)
    
    def get_normalized_rank_by_pp(self) -> dict[str, float]:
        """
        Depending on direction of priority policy, min-max-normalized "rank" for each buffered object type is determined based on its average number of objects per event. The type with the smallest rank is most likely to be removed.

        Returns
        -------
        dict[str, float]
            Mapping of object types to min-max-normalized ranks in [0, 1].
        """
        key_to_val = dict()
        for ot in self.buf:
            # No "+1" to avoid division by 0 for PrioPolicyOrder.MAX necessary since #objects per event per OT is at least 1
            if self.prio_order == PrioPolicyOrder.MIN:
                key_to_val[ot] = np.mean(self.buf[ot])
            else:
                key_to_val[ot] = 1/np.mean(self.buf[ot])

        return min_max_normalize_dict(key_to_val)
    
    def to_string(self) -> str:
        """
        Creates string describing priority-policy buffer including its parameters (e.g. direction) and buffered object types in tabular format.

        Returns
        -------
        str
            Output string describing parameters and content of buffer.
        """
        ret_str = f'Priority-policy buffer characteristics:\n - priority policy: {self.pp.value}\n - most likely to get removed for {self.prio_order.value} value\n - window size: {self.window_size}\n - buffer size: {len(self)}\n'

        # Unroll buffer structure: OT -> [#objects per event, ...]
        avg_num_obj_key = 'avg # objects per event'
        num_events_key = '# buffered events'
        unrolled_buf = {OBJECT_TYPE: list(), num_events_key: list(), avg_num_obj_key : list()}

        for ot, event_list in self.buf.items():
            unrolled_buf[OBJECT_TYPE].append(ot)
            unrolled_buf[num_events_key].append(len(event_list))
            unrolled_buf[avg_num_obj_key].append(np.mean(event_list))

        buf_df = pd.DataFrame.from_dict(unrolled_buf)
        ret_str += tabulate(buf_df, headers='keys', tablefmt='psql', showindex=True)
        ret_str += '\n'

        return ret_str
    

class PPBEventsPerObjectType(PrioPolicyBuffer):
    """
    Represents a priority-policy buffer that counts the number of events per object type.

    Attributes
    ----------
    prio_order : PrioPolicyOrder
        Direction of priority policy.
    max_counter : int
        Maximum counter for number of events for any object type before all counters are reset to avoid overflow.
    pp : PriorityPolicy
        Enum of corresponding priority policy.
    buf_name : str
        Name and direction of priority policy.
    buf : dict[str, int]
        Dictionary containing priority-policy buffer items, i.e. mapping object types to number of events.
    """

    def __init__(self, prio_order : PrioPolicyOrder, max_counter : int = 100000):
        """
        Initializes a PPBEventsPerObjectType object.

        Parameters
        ----------
        prio_order : PrioPolicyOrder
            Direction of priority policy.
        max_counter : int, default=100000
            Maximum counter for number of events for any object type before all counters are reset.
        """
        self.prio_order = prio_order
        self.max_counter = max_counter
        
        self.pp = PriorityPolicy.EVENTS_PER_OT
        self.buf_name = f'{prio_order.value} {self.pp.value}'
        # OT -> #events
        self.buf : dict[str, int] = dict()

    def __len__(self) -> int:
        return len(self.buf)

    def update(self, stream_item : Event) -> None:
        """
        Updates number of events for currently buffered object types or adds new types.

        Parameters
        ----------
        stream_item : Event
            In-coming event whose involved object types update the buffer.

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
            If unsupported stream item is given.
        """
        unique_ots = set([d['objectType'] for d in stream_item.e2o_relations])

        # Increase #events per OT
        for new_ot in unique_ots:
            if new_ot not in self.buf:
                self.buf[new_ot] = 1
            else:
                new_count = self.buf[new_ot] + 1
                # Do hard reset of all event counters per OT once single reaches max. counter
                if new_count >= self.max_counter:
                    new_count = 1
                    for new_ot in self.buf:
                        self.buf[new_ot] = 1
                self.buf[new_ot] = new_count
    
    def get_normalized_rank_by_pp(self) -> dict[str, float]:
        """
        Depending on direction of priority policy, min-max-normalized "rank" for each buffered object type is determined based on its total number of events. The type with the smallest rank is most likely to be removed.

        Returns
        -------
        dict[str, float]
            Mapping of object types to min-max-normalized ranks in [0, 1].
        """
        # No "+1" correction of values necessary to avoid division by 0 in case of PrioPolicyOrder.MAX since #events per OT is at least 1
        if self.prio_order == PrioPolicyOrder.MIN:
            key_to_val = self.buf.copy()
        else:
            key_to_val = {key: 1/val for key, val in self.buf.items()}

        return min_max_normalize_dict(key_to_val)
    
    def to_string(self) -> str:
        """
        Creates string describing priority-policy buffer including its parameters (e.g. direction) and buffered object types in tabular format.

        Returns
        -------
        str
            Output string describing parameters and content of buffer.
        """
        ret_str = f'Priority-policy buffer characteristics:\n - priority policy: {self.pp.value}\n - most likely to get removed for {self.prio_order.value} value\n - max counter: {self.max_counter}\n - buffer size: {len(self)}\n'

        # Unroll buffer structure: OT -> #events
        num_events_key = '# events'
        unrolled_buf = {OBJECT_TYPE: list(), num_events_key: list()}

        for ot, num_events in self.buf.items():
            unrolled_buf[OBJECT_TYPE].append(ot)
            unrolled_buf[num_events_key].append(num_events)

        buf_df = pd.DataFrame.from_dict(unrolled_buf)
        ret_str += tabulate(buf_df, headers='keys', tablefmt='psql', showindex=True)
        ret_str += '\n'

        return ret_str


class PPBCustom(PrioPolicyBuffer):
    """
    Represents a priority-policy buffer based on a custom ranking of object types, if they are known beforehand.

    Attributes
    ----------
    prio_order : PrioPolicyOrder
        Direction of priority policy.
    pp : PriorityPolicy
        Enum of corresponding priority policy.
    ot_to_rank : dict[str, float]
        Mapping of object types to static, min-max-normalized ranks.
    buf_name : str
        Name and direction of priority policy.
    """

    def __init__(self, prio_order : PrioPolicyOrder, ot_list : list[str]):
        """
        Initializes a PPBCustom object.

        Parameters
        ----------
        prio_order : PrioPolicyOrder
            Direction of priority policy.
        ot_list : list[str]
            List of all object types, sorted in order of increasing or decreasing rank, depending on the direction of the priority policy.
        """
        self.prio_order = prio_order
        
        # Translate order of OTs to linear ranking of OTs once
        if self.prio_order == PrioPolicyOrder.MIN:
            ot_to_rank = {ot: i for i, ot in enumerate(ot_list)}
        # Add "index+1" as value to avoid division by 0 in case of PrioPolicyOrder.MAX
        else:
            ot_to_rank = {ot: 1/(i+1) for i, ot in enumerate(ot_list)}
        
        self.ot_to_rank = min_max_normalize_dict(ot_to_rank)
        self.pp = PriorityPolicy.CUSTOM_OT_ORDER
        self.buf_name = f'{prio_order.value} {self.pp.value}'

    def __len__(self) -> int:
        return len(self.ot_to_rank)

    def update(self, stream_item : Union[Event, O2OUpdate]) -> None:
        pass
    
    def get_normalized_rank_by_pp(self) -> dict[str, float]:
        """
        Depending on direction of priority policy, the min-max-normalized "rank" for each buffered object type refers to its position in the ranking used to initialize the PPB object.

        Returns
        -------
        dict[str, float]
            Mapping of object types to min-max-normalized ranks in [0, 1].
        """
        return self.ot_to_rank
    
    def to_string(self) -> str:
        """
        Creates string describing priority-policy buffer including its parameters (e.g. direction) and order of object types.

        Returns
        -------
        str
            Output string describing parameters and content of buffer.
        """
        ret_str = f'Priority-policy buffer characteristics:\n - priority policy: {self.pp.value}\n - most likely to get removed for min rank\n - buffer size: {len(self)}\n'

        # Unroll buffer structure: OT -> #events
        rank_key = 'min-max-normalized rank'
        unrolled_buf = {OBJECT_TYPE: list(), rank_key: list()}

        for ot, rank in self.ot_to_rank.items():
            unrolled_buf[OBJECT_TYPE].append(ot)
            unrolled_buf[rank_key].append(rank)

        buf_df = pd.DataFrame.from_dict(unrolled_buf)
        ret_str += tabulate(buf_df, headers='keys', tablefmt='psql', showindex=True)
        ret_str += '\n'

        return ret_str


class PPBObjectsPerObjectType(PrioPolicyBuffer):
    """
    Represents a priority-policy buffer that counts the number of objects per object type.

    Attributes
    ----------
    prio_order : PrioPolicyOrder
        Direction of priority policy.
    window_size : int
        Maximum number of unique objects that are buffered per object type.
    pp : PriorityPolicy
        Enum of corresponding priority policy.
    buf_name : str
        Name and direction of priority policy.
    buf : dict[str, set[str]]
        Dictionary containing priority-policy buffer items, i.e. mapping object types to number to set of unique, observed objects.
    """
    
    def __init__(self, prio_order : PrioPolicyOrder, window_size : int = 1000):
        """
        Initializes a PPBObjectsPerObjectType object.

        Parameters
        ----------
        prio_order : PrioPolicyOrder
            Direction of priority policy.
        window_size : int, default=1000
            Maximum number of unique objects that are buffered per object type before all buffered objects are reset for all object types to ensure limited memory use.
        """
        self.prio_order = prio_order
        self.window_size = window_size
        
        self.pp = PriorityPolicy.OBJ_PER_OT
        self.buf_name = f'{prio_order.value} {self.pp.value}'
        # OT -> set of objects of limited length
        self.buf : dict[str, set[str]] = dict()

    def __len__(self) -> int:
        return len(self.buf)

    def update(self, stream_item : Union[Event, O2OUpdate]) -> None:
        """
        Updates set of unique objects for currently buffered object types or adds new types.

        Parameters
        ----------
        stream_item : Union[Event, O2OUpdate]
            In-coming event or object-to-object update whose involved objects update the buffer.

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
            If unsupported stream item is given.
        """
        if isinstance(stream_item, Event):
            new_tups = [(d['objectType'], d['objectId']) for d in stream_item.e2o_relations]
        elif isinstance(stream_item, O2OUpdate):
            new_tups = [(stream_item.type, stream_item.id), (stream_item.target_type, stream_item.target_id)]
        else:
            raise NotImplementedError(f'#Objects-per-OT-based PriorityPolicy buffer not implemented for given stream item of type {type(stream_item)}.')
        
        for new_ot_key, new_oid in new_tups:
            if new_ot_key not in self.buf:
                self.buf[new_ot_key] = set([new_oid])
            else:
                self.buf[new_ot_key].add(new_oid)
                # Do hard reset of all object sets once single OT reaches maximum window size w.r.t. #unique objects
                if len(self.buf[new_ot_key]) >= self.window_size:
                    for buffered_ot in self.buf:
                        self.buf[buffered_ot] = set([self.buf[buffered_ot].pop()])
                    self.buf[new_ot_key] = set([new_oid])
    
    def get_normalized_rank_by_pp(self) -> dict[str, float]:
        """
        Depending on direction of priority policy, min-max-normalized "rank" for each buffered object type is determined based on its number of unique objects. The type with the smallest rank is most likely to be removed.

        Returns
        -------
        dict[str, float]
            Mapping of object types to min-max-normalized ranks in [0, 1].
        """
        # No "+1" correction of #unique OIDs necessary to avoid division by zero in case of PrioPolicyOrder.MAX since #unique OIDs per OT is at least 1
        if self.prio_order == PrioPolicyOrder.MIN:
            key_to_val = {ot: len(unique_oids) for ot, unique_oids in self.buf.items()}
        else:
            key_to_val = {ot: 1/len(unique_oids) for ot, unique_oids in self.buf.items()}

        return min_max_normalize_dict(key_to_val)
    
    def to_string(self) -> str:
        """
        Creates string describing priority-policy buffer including its parameters (e.g. direction) and buffered object types in tabular format.

        Returns
        -------
        str
            Output string describing parameters and content of buffer.
        """
        ret_str = f'Priority-policy buffer characteristics:\n - priority policy: {self.pp.value}\n - most likely to get removed for {self.prio_order.value} value\n - window size: {self.window_size}\n - buffer size: {len(self)}\n'

        # Unroll buffer structure: OT -> set of objects of limited length
        num_obj_key = '# unique objects'
        unrolled_buf = {OBJECT_TYPE: list(), num_obj_key: list()}

        for ot, obj_set in self.buf.items():
            unrolled_buf[OBJECT_TYPE].append(ot)
            unrolled_buf[num_obj_key].append(len(obj_set))

        buf_df = pd.DataFrame.from_dict(unrolled_buf)
        ret_str += tabulate(buf_df, headers='keys', tablefmt='psql', showindex=True)
        ret_str += '\n'

        return ret_str