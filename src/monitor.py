"""\
Classes for collecting cache and runtime information during stream processing for evaluation.
__author__: "Nina Löseke"
"""

from typing import Union
import pandas as pd
from cache_policy_buffers import Buffer, BufferOfDicts, BufferOfDictLists
from vars import *
from utils import Event, O2OUpdate, ObjectAttributeUpdate


class RuntimeMonitor(object):
    """
    Collects runtime data during stream processing.

    Attributes
    ----------
    buf_update_times : list
        Recorded runtime information about each model-buffer update during stream processing.
    buf_update_times_df : pd.DataFrame
        `buf_update_times` represented as DataFrame.
    item_processing_times : list
        Recorded runtime information about each processed stream item.
    item_processing_times_df : pd.DataFrame
        `item_processing_times` represented as DataFrame.
    total_stream_processing_time : float
        Total time it takes to process stream.
    stream_item_count : int
        Count of processed stream items that is increased during stream processing.
    """
    
    def __init__(self) -> None:
        """Initializes empty RuntimeMonitor object."""
        self.buf_update_times = list()
        self.buf_update_times_df = None
        self.item_processing_times = list()
        self.item_processing_times_df = None
        self.total_stream_processing_time = None
        
        # Set up data structure to track information during stream processing on stream-item basis
        self.stream_item_count = 0
    
    def increase_stream_item_count(self) -> None:
        """
        Increases count of stream items by one; triggered when a new in-coming stream item is processed.

        Returns
        -------
        None
        """
        self.stream_item_count += 1

    def set_stream_item_count(self, count : int) -> None:
        """
        Overwrites stream-item count with given value.

        Parameters
        ----------
        count : int
            New stream-item count.
        
        Returns
        -------
        None
        """
        self.stream_item_count = count

    def register_buf_update_time(self, ns : float, buf_name : str) -> None:
        """
        Logs new information about buffer-update time as tuple of current stream-item count, time [ns], and name of model buffer.

        Parameters
        ----------
        ns : float
            Time [ns] it took to update model buffer.
        buf_name : str
            Name of model buffer.

        Returns
        -------
        None
        """
        self.buf_update_times.append([self.stream_item_count, ns, buf_name])
    
    def register_item_processing_time(self, ns : float, item_type : str) -> None:
        """
        Logs new information about item-processing time as tuple of current stream-item count, time [ns], and type of stream item.

        Parameters
        ----------
        ns : float
            Time [ns] it took to process stream item.
        item_type : str
            Type of stream item, e.g. Event.

        Returns
        -------
        None
        """
        self.item_processing_times.append([self.stream_item_count, ns, item_type])

    def register_total_stream_processing_time(self, ns : float) -> None:
        """
        Logs total stream-processing time [ns].

        Parameters
        ----------
        ns : float
            Time [ns] it took to process entire object-centric event stream.

        Returns
        -------
        None
        """
        self.total_stream_processing_time = ns
    
    def create_dataframes(self) -> None:
        """Converts collected information into DataFrames."""
        self.buf_update_times_df = pd.DataFrame(columns=[M_STREAM_ITEM, M_BUF_UPDATE_TIME, M_BUF_NAME], data=self.buf_update_times)
        # Transform model-buffer update times from ns into ms
        self.buf_update_times_df[M_BUF_UPDATE_TIME] = self.buf_update_times_df[M_BUF_UPDATE_TIME].apply(lambda x: x/(10**6))

        self.item_processing_times_df = pd.DataFrame(columns=[M_STREAM_ITEM, M_ITEM_PROCESSING_TIME, M_ITEM_TYPE], data=self.item_processing_times)
        # Transform item-processing times from ns into ms
        self.item_processing_times_df[M_ITEM_PROCESSING_TIME] = self.item_processing_times_df[M_ITEM_PROCESSING_TIME].apply(lambda x: x/(10**6))
        # Transform total stream-processing time from ns to ms
        self.total_stream_processing_time = self.total_stream_processing_time / 10**6


class CacheMonitor(object):
    """
    Collects cache-behavior data during stream processing.

    Attributes
    ----------
    total_model_buf_sizes : int
        Sum of model-buffer sizes.
    stream_item_count : int
        Count of processed stream items that is increased during stream processing.
    stream_size : int
        Total length of stream.
    eval_stream_percentages : list
        Stream-item counts marking 10%, 20%, ..., 100% of entire stream.
    hits_df : pd.DataFrame
        DataFrame with recorded cache hits per object type and model buffer.
    hits : list
        List corresponding to `hits_df`.
    evic_df : pd.DataFrame
        DataFrame with recorded cache evictions per object type and model buffer.
    evic : list
        List corresponding to `evic_df`.
    full_evic_df : pd.DataFrame
        DataFrame with recorded complete cache evictions per object type and model buffer.
    full_evic : list
        List corresponding to `full_evic_df`.
    ot_frac_df : pd.DataFrame
        DataFrame with fraction of object types per model buffer at different stages in stream.
    ot_frac : list
        List corresponding to `ot_frac_df`.
    ppb_size_df : pd.DataFrame
        DataFrame with recorded size of priority-policy buffer.
    ppb_size : list
        List corresponding to `ppb_size_df`.
    """
    
    def __init__(self) -> None:
        """Initializes empty CacheMonitor object."""
        # Set up data structures to track information during stream processing
        self.total_model_buf_sizes = None
        self.stream_item_count = 0
        self.stream_size = 0
        self.eval_stream_percentages = list()

        self.hits_df = None
        self.hits = list()
        self.evic_df = None
        self.evic = list()
        self.full_evic_df = None
        self.full_evic = list()
        self.ot_frac_df = None
        self.ot_frac = list()
        self.ppb_size_df = None
        self.ppb_size = list()

    def set_total_buf_sizes(self, total_model_buf_sizes : int) -> None:
        """
        Sets sum of model-buffer sizes to given value; used to compare priority-policy buffer sizes against.

        Parameters
        ----------
        total_model_buf_sizes : int
            Sum of model-buffer sizes.

        Returns
        -------
        None
        """
        self.total_model_buf_sizes = total_model_buf_sizes
    
    def register_stream_size(self, stream : list[Union[Event, O2OUpdate, ObjectAttributeUpdate]]) -> None:
        """
        Saves total length of stream to be processed and derives stream-item counts corresponding to 10%, 20%, ..., 100% marks in stream.

        Parameters
        ----------
        stream : list[Union[Event, O2OUpdate, ObjectAttributeUpdate]]
            Object-centric event stream whose size is recorded.
        
        Returns
        -------
        None
        """
        self.stream_size = len(stream)

        # Derive stream-item counts at which to evaluation OT fractions per buffer (0%, 10%, 20% etc. of stream)
        self.eval_stream_percentages = [int(self.stream_size*pct/100) for pct in range(0, 110, 10)]
    
    def increase_stream_item_count(self) -> None:
        """
        Increases count of stream items by one; triggered when a new in-coming stream item is processed.

        Returns
        -------
        None
        """
        self.stream_item_count += 1
    
    def set_stream_item_count(self, count : int) -> None:
        """
        Overwrites stream-item count with given value.

        Parameters
        ----------
        count : int
            New stream-item count.

        Returns
        -------
        None
        """
        self.stream_item_count = count

    def check_for_ot_frac_per_buffer(self) -> bool:
        """
        Returns True if current-stream item count corresponds to a 10% mark in processed stream.

        Returns
        -------
        bool
        """
        if self.stream_item_count in self.eval_stream_percentages:
            return True
        return False
    
    def update_ppb_size(self, ppb_size : int) -> None:
        """
        Logs size of priority-policy buffer at current stream-item count.

        Returns
        -------
        None
        """
        self.ppb_size.append([self.stream_item_count, ppb_size])

    def register_ot_frac_for_single_ot_buffers(self, ot_to_buf : dict[str, Buffer]) -> None:
        """
        Logs fractions of object types in separate model buffers per object type.

        Parameters
        ----------
        ot_to_buf : dict[str, Buffer]
            Mapping of object types to model buffers.

        Returns
        -------
        None
        """
        res_tups = list()
        for ot, buf in ot_to_buf.items():
            res_tups.append((ot, len(buf), buf.buf_name))
        num_ot_occurrences = sum([tup[1] for tup in res_tups])
        for ot, total_ot_count, buf_name in res_tups:
            self.ot_frac.append([self.stream_item_count, total_ot_count/num_ot_occurrences, ot, buf_name])
    
    def register_ot_frac_for_mixed_buffer(self, buf : Buffer, nested_ot_key : str = None, ot_tup_as_buf_key : bool = False) -> None:
        """
        Logs fractions of object types in model buffer that is shared by all object types.

        Parameters
        ----------
        buf : Buffer
            Model buffer that is shared by all object types.
        nested_ot_key : str, default=None
            Value-dictionary key to access object type of buffer item, if necessary.
        ot_tup_as_buf_key : bool, default=False
            Indicates if keys of buffer items correspond to type tuples.

        Returns
        -------
        None
        """
        ot_to_count = dict()
        buf_name = buf.buf_name
        for buf_key, buf_val in buf.buf.items():
            if nested_ot_key is not None:
                if isinstance(buf, BufferOfDicts):
                    ot = buf_val[nested_ot_key]
                    ot_to_count[ot] = ot_to_count.get(ot, 0) + 1
                elif isinstance(buf, BufferOfDictLists):
                    ots = [buf_dict[nested_ot_key] for buf_dict in buf_val]
                    for ot in ots:
                        ot_to_count[ot] = ot_to_count.get(ot, 0) + 1
            elif ot_tup_as_buf_key:
                ots = set([buf_key[0], buf_key[1]])
                ots.discard(None)
                assert isinstance(buf, BufferOfDictLists)
                for ot in ots:
                    # Associate each OT in buffer key w/ length of associated list of dicts
                    ot_to_count[ot] = ot_to_count.get(ot, 0) + len(buf_val)
            else:
                raise RuntimeError(f'Must specify one option to determine how to register OT fraction for buffer {buf.buf_name} via CacheMonitor object!')

        num_ot_occurrences = sum(ot_to_count.values())
        for ot in ot_to_count:
            self.ot_frac.append([self.stream_item_count, ot_to_count[ot]/num_ot_occurrences, ot, buf_name])

    def register_buf_eviction(self, evicted_ot : str, full_eviction : bool, buf_name : str) -> None:
        """
        Logs new (complete) eviction of a given object type for a certain model buffer.

        Parameters
        ----------
        evicted_ot : str
            Object type of buffer item that was removed from model buffer.
        full_eviction : bool
            Indicates if object type was fully removed from model buffer or still has some buffered items.
        buf_name : str
            Name of corresponding model buffer.

        Returns
        -------
        None
        """
        self.evic.append([self.stream_item_count, evicted_ot, buf_name])

        if full_eviction:
            self.full_evic.append([self.stream_item_count, evicted_ot, buf_name])

    def register_buf_insertion(self, num_hits : int, hit_miss_ot : str, buf_name : str) -> None:
        """
        Logs new cache hit/miss for newly inserted buffer item and its associated object type and model buffer.

        Parameters
        ----------
        num_hits : int
            Indicates number of hits for newly added buffer item (0 or 1) based on buffer-item key.
        hit_miss_ot : str
            Object type for which new buffer item was inserted.
        buf_name : str
            Name of corresponding model buffer.

        Returns
        -------
        None
        """
        self.hits.append([self.stream_item_count, num_hits, 1-num_hits, hit_miss_ot, buf_name])

    def create_dataframes(self) -> None:
        """Converts collected information into DataFrames."""
        self.hits_df = pd.DataFrame(columns=[M_STREAM_ITEM, M_CACHE_HITS, M_CACHE_MISSES, M_OBJECT_TYPE, M_BUF_NAME], data=self.hits)
        self.evic_df = pd.DataFrame(columns=[M_STREAM_ITEM, M_OBJECT_TYPE, M_BUF_NAME], data=self.evic)
        self.full_evic_df = pd.DataFrame(columns=[M_STREAM_ITEM, M_OBJECT_TYPE, M_BUF_NAME], data=self.full_evic)
        self.ot_frac_df = pd.DataFrame(columns=[M_STREAM_ITEM, M_OT_PERCENTAGE, M_OBJECT_TYPE, M_BUF_NAME], data=self.ot_frac)

        # Aggregate PPB sizes over potential (sub-)buffers w/ separate PPB (e.g. 1 PPB maintained per DFG in OCPN using OcdfgBufferPerObjectType)
        self.ppb_size_df = pd.DataFrame(columns=[M_STREAM_ITEM, M_PPB_SIZE], data=self.ppb_size)
        self.ppb_size_df = self.ppb_size_df.groupby(M_STREAM_ITEM).aggregate({M_PPB_SIZE: 'sum'}).reset_index()