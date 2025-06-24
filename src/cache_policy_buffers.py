"""\
Classes representing two different types of underlying model buffers.
__author__: "Nina Löseke"
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple
from enum import Enum
import random
import pandas as pd
from tabulate import tabulate
import itertools
from vars import *
from utils import min_max_normalize_dict


class CachePolicy(Enum):
    """
    Represents supported cache-replacement policies used by model buffers.
    """

    FIFO = "FIFO"
    LRU = "LRU"
    LFU = "LFU"
    LFU_DA = "LFU-DA"   # "least frequently used, dynamic ageing"
    RR = "RR"           # "random replacement"


class Buffer(ABC):
    """
    Abstract base class defining required functionality of a model buffer.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def is_full(self):
        pass
    
    @abstractmethod
    def __len__(self):
        pass
    
    @abstractmethod
    def to_string(self):
        pass
    
    @abstractmethod
    def get_normalized_rank_by_cp(self):
        pass

    @abstractmethod
    def get_buffered_nested_values_for_key(self):
        pass
    
    @abstractmethod
    def get_min_max_normalized_key_value_pairs(self):
        pass

    @abstractmethod
    def insert_new_item_by_cp(self):
        pass

    @abstractmethod
    def get_removal_key_by_cp(self):
        pass

    @abstractmethod
    def remove_item(self):
        pass


class BufferOfDicts(Buffer):
    """
    Represents a generic model buffer whose buffer items are buffered inside a dictionary mapping each unique key to a single dictionary.

    Attributes
    ----------
    buf_size : int
        Maximum number of buffered items, i.e. key-value pairs.
    cp : CachePolicy
        Cache policy used to replace buffer items when full.
    buf_name : str
        Name of buffer
    buf_key_name : str
        Name of items used as buffer keys, e.g. object IDs.
    ot : str
        Object type to which all buffer items belong; if None, buffer items of any object type are permitted.
    max_counter : int
        Maximum counter value (e.g. for key frequency or cache age) before reset.
    buf_cache_age : int
        Cache age that is increased with every buffer item removed from a full buffer.
    buf : dict
        Dictionary containing buffer items, i.e. mapping unique key to a value dictionary.
    """

    def __init__(self, buf_size : int, cp : CachePolicy, buf_name : str, buf_key_name : str, ot : str = None, max_counter : int = 100000):
        """
        Initializes a BufferOfDicts object.

        Parameters
        ----------
        buf_size : int
            Maximum number of buffered items, i.e. key-value pairs.
        cp : CachePolicy
            Cache policy used to replace buffer items when full.
        buf_name : str
            Name of buffer.
        buf_key_name : str
            Name of items used as buffer keys, e.g. object IDs.
        ot : str, default=None
            Object type to which all buffer items belong; if None, buffer items of any object type are permitted.
        max_counter : int, default=100000
            Maximum counter value (e.g. for key frequency or cache age) before reset.
        """
        self.buf_size = buf_size
        self.cp = cp
        self.buf_name = buf_name
        self.buf_key_name = buf_key_name
        # Used by DFG buffers
        self.ot = ot
        self.max_counter = max_counter

        self.buf_cache_age = 0
        self.buf = dict()
    
    def reduce_to_oids_ots(self, oids_ots_keep : set[str], nested_key : str = None) -> None:
        """
        Remove buffered items not related to the specified set of object IDs or object types.

        Parameters
        ----------
        oid_ots_keep : set[str]
            Set of object IDs or object types to keep.
        nested_key : str, default=None
            Specifies value-dictionary key to access object ID or object type of a buffer item, if necessary.

        Returns
        -------
        None
        """
        if nested_key is None:
            for buf_key in list(self.buf.keys()):
                if buf_key not in oids_ots_keep:
                    self.remove_item(buf_key)
        else:
            for buf_key, buf_dict in list(self.buf.items()):
                if buf_dict[nested_key] not in oids_ots_keep:
                    self.remove_item(buf_key)
    
    def get_normalized_rank_by_cp(self) -> dict[Any, float]:
        """
        Create ranking of buffer items based on cache policy of model buffer.

        Returns
        -------
        dict[Any, float]
            Dictionary mapping each buffered key to a min-max-normalized rank where lowest rank indicates item most likely to be removed according to cache policy.
        
        Raises
        ------
        NotImplementedError
            If cache policy of model buffer is unknown.
        """
        key_to_val = dict()

        if self.cp == CachePolicy.FIFO:
            for i, buf_key in enumerate(self.buf):
                key_to_val[buf_key] = i
        
        elif self.cp == CachePolicy.LRU:
            max_last_seen = max([buf_dict[LAST_SEEN] for buf_dict in self.buf.values()])
            # Assign smallest numerical value to buffer key w/ most recent "last-seen" timestamp; maintain distance between last-seen timestamps of buffer keys
            key_to_val = {buf_key: (buf_dict[LAST_SEEN]-max_last_seen).total_seconds() for buf_key, buf_dict in self.buf.items()}
        
        elif self.cp == CachePolicy.LFU:
            key_to_val = self.get_min_max_normalized_key_value_pairs(FREQUENCY)
        
        elif self.cp == CachePolicy.LFU_DA:
            key_to_freq = self.get_min_max_normalized_key_value_pairs(FREQUENCY)
            key_to_cache_age = self.get_min_max_normalized_key_value_pairs(CACHE_AGE)
            
            for buf_key_id in key_to_cache_age:
                key_to_val[buf_key_id] = key_to_cache_age[buf_key_id] + key_to_freq[buf_key_id]
        
        elif self.cp == CachePolicy.RR:
            # Shuffle buffer keys to create random ranking
            buf_keys = list(self.buf.keys())
            random.shuffle(buf_keys)
            for i in range(len(buf_keys)):
                key_to_val[buf_keys[i]] = i
        
        else:
            raise NotImplementedError(f'Creating ranking for buffered items not implemented for given cache policy: {self.cp}')

        return min_max_normalize_dict(key_to_val)
    
    def get_min_max_normalized_key_value_pairs(self, key_in_dict : Any) -> dict[Any, float]:
        """
        Extract pairs of buffer-items keys and min-max-normalized values for a given value-dictionary key.

        Parameters
        ----------
        key_in_dict : Any
            Value-dictionary key for which to extract values for all buffer items.

        Returns
        -------
        dict[Any, float]
            Dictionary mapping each buffer-item key to single, min-max-normalized value.
        """
        key_to_val = dict()
        for buf_key, buf_dict in self.buf.items():
            key_to_val[buf_key] = buf_dict[key_in_dict]
        
        return min_max_normalize_dict(key_to_val)

    def get_removal_key_by_cp(self) -> Any:
        """
        Determine key of buffer item to remove from full model buffer according to cache policy.

        Returns
        -------
        Any
            Buffer key of buffer item to remove according to cache policy, e.g. object ID.
        
        Raises
        ------
        NotImplementedError
            If cache policy of model buffer is unknown.
        """
        if self.cp == CachePolicy.FIFO:
            key_pop = next(iter(self.buf))
        
        elif self.cp == CachePolicy.LRU:
            key_pop = min(self.buf.items(), key=lambda buf_tup: buf_tup[1][LAST_SEEN])[0]
        
        elif self.cp == CachePolicy.LFU:
            key_pop = min(self.buf.items(), key=lambda buf_tup: buf_tup[1][FREQUENCY])[0]
        
        elif self.cp == CachePolicy.LFU_DA:
            key_to_freq = self.get_min_max_normalized_key_value_pairs(FREQUENCY)
            key_to_cache_age = self.get_min_max_normalized_key_value_pairs(CACHE_AGE)
            
            # Translate ranking of buffer keys according to frequency and cache age into value per key
            key_to_rank = dict()
            for buf_key in key_to_cache_age:
                key_to_rank[buf_key] = key_to_cache_age[buf_key] + key_to_freq[buf_key]

            # The lower the frequency and the lower the cache age, the more likely to be removed
            key_pop = min(key_to_rank.items(), key=lambda key_tup: key_tup[1])[0]

            # Increase cache age whenever there is a removal
            self.buf_cache_age += 1

            # Reset cache ages in buffer once max. counter is reached
            if self.buf_cache_age >= self.max_counter:
                for buffered_key in self.buf:
                    self.buf[buffered_key][CACHE_AGE] = 0
                self.buf_cache_age = 0
        
        elif self.cp == CachePolicy.RR:
            key_id_pop = random.randint(0, self.buf_size-1)
            key_pop = list(self.buf.keys())[key_id_pop]

        else:
            raise NotImplementedError(f'Buffer-item removal not implemented for given cache policy: {self.cp}')
    
        return key_pop
    
    def remove_item(self, key_pop : Any) -> None:
        """
        Removes buffer item with given key from `buf` dictionary that holds buffered items.

        Parameters
        ----------
        key_pop : Any
            Key of buffered item to remove.

        Returns
        -------
        None
        """
        self.buf.pop(key_pop)

    def insert_new_item_by_cp(self, new_key : Any, new_dict_overwrite : dict[str, Any], new_dict_init : dict[str, Any] = dict()) -> None:
        """
        Adds given buffer item to buffer and replaces existing item according to cache policy if buffer is full.

        Parameters
        ----------
        new_key : Any
            Key of new buffer item, e.g. object ID.
        new_dict_overwrite : dict[str, Any]
            Key-value pairs to overwrite in value dictionary if `new_key` is already buffered.
        new_dict_init : dict[str, Any], default={}
            Key-value pairs to initialize in value dictionary once upon insertion of new buffer item.

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
            If cache policy of model buffer is unknown.
        """
        # Update frequency for buffer key at time of insertion since it may be removed during insertion of earlier buffer items for same stream item if buffer is full
        if self.cp in [CachePolicy.LFU, CachePolicy.LFU_DA]:
            new_key_freq = 1
            if new_key in self.buf:
                new_key_freq += self.buf[new_key][FREQUENCY]
                # If frequency count for single buffer key reaches maximum, do hard reset
                if new_key_freq >= self.max_counter:
                    for buffered_key in self.buf:
                        self.buf[buffered_key][FREQUENCY] = 1
                    new_key_freq = 1
            new_dict_overwrite[FREQUENCY] = new_key_freq

        # Cache age for LFU-DA can only be determined at time of insertion
        if self.cp == CachePolicy.LFU_DA:
            new_dict_init[CACHE_AGE] = self.buf_cache_age
            
        if self.cp in [CachePolicy.FIFO, CachePolicy.LFU, CachePolicy.LFU_DA, CachePolicy.RR]:
            if new_key in self.buf:
                for ow_key, ow_value in new_dict_overwrite.items():
                    self.buf[new_key][ow_key] = ow_value
            else:
                self.buf[new_key] = new_dict_init | new_dict_overwrite
        
        elif self.cp == CachePolicy.LRU:
            # Re-append already-seen buffer key w/ original "initialization" values to maintain LRU order via buffer keys
            if new_key in self.buf:
                new_dict_init = self.buf[new_key]
                self.buf.pop(new_key)
            self.buf[new_key] = new_dict_init | new_dict_overwrite
        
        else:
            raise NotImplementedError(f'Adding new buffer item based not implemented for given cache policy: {self.cp}')
    
    def get_buffered_nested_values_for_key(self, nested_key : str) -> set[Any]:
        """
        Returns set of values for given key in buffered value dictionaries.

        Parameters
        ----------
        nested_key : str
            Key in value dictionary for which unique values are extracted.
        
        Returns
        -------
        set[Any]
            Set of values for given nested key.
        """
        return set([buf_dict[nested_key] for buf_dict in self.buf.values()])

    def is_full(self) -> bool:
        """
        Checks if model buffer is full.

        Returns
        -------
        bool
            True if buffer is full, otherwise False.
        """
        return len(self) == self.buf_size

    def __len__(self) -> int:
        """
        Returns length of buffer in terms of the number of buffered items.

        Returns
        -------
        int
            Number of buffered items.
        """
        return len(self.buf)

    def to_string(self, reorder_cols : list[str] = None) -> str:
        """
        Creates string describing buffer including its parameters (cache policy etc.) and buffered items in tabular format.

        Parameters
        ----------
        reorder_cols : list[str], default=None
            List of column names referring to names of buffer-item keys and value-dictionary keys that force order of columns in tabular output.
        
        Returns
        -------
        str
            Output string describing parameters and content of buffer.
        """
        ret_str = f'{self.cp.value} {self.buf_name} characteristics:\n - buffer size: {self.buf_size}\n - max counter: {self.max_counter}\n - object type: {self.ot if self.ot else "--"}\n'

        # Additional transformation for buffers to turn key->dict items into "unrolled" DataFrame rows
        unrolled_buf = {self.buf_key_name: list()} if not reorder_cols else {col_name: list() for col_name in reorder_cols}
        for key, val in self.buf.items():
            if isinstance(val, dict):
                unrolled_buf[self.buf_key_name].append(key)
                for val_key, val_val in val.items():
                    if val_key not in unrolled_buf:
                        unrolled_buf[val_key] = list()
                    unrolled_buf[val_key].append(val_val)
            
            else:
                raise NotImplementedError(f'Buffer unrolling for tabular pretty-printing not supported! Got unexpected {type(val)} as data type associated w/ buffer keys.')
        
        buf_df = pd.DataFrame.from_dict(unrolled_buf)
        if reorder_cols is not None:
            buf_df = buf_df[reorder_cols]

        # Source: https://stackoverflow.com/questions/18528533/pretty-printing-a-pandas-dataframe
        ret_str += tabulate(buf_df, headers='keys', tablefmt='psql', showindex=True)
        ret_str += '\n'

        return ret_str


class BufferOfDictLists(Buffer):
    """
    Represents a generic model buffer whose buffer items are maintained in a dictionary mapping each unique key to a list of value dictionaries.
    Each value dictionary corresponds to a single buffer item.

    Attributes
    ----------
    buf_size : int
        Maximum number of buffered items, i.e. key-value pairs.
    cp : CachePolicy
        Cache policy used to replace buffer items when full.
    buf_name : str
        Name of buffer
    buf_key_name : str
        Name of items used as buffer keys, e.g. object IDs.
    ot : str
        Object type to which all buffer items belong; if None, buffer items of any object type are permitted.
    max_counter : int
        Maximum counter value (e.g. for key frequency or cache age) before reset.
    buf_cache_age : int
        Cache age that is increased with every buffer item removed from a full buffer.
    buf : dict
        Dictionary containing buffer items, i.e. mapping unique key to a value dictionary.
    """
    
    def __init__(self, buf_size : int, cp : CachePolicy, buf_name : str, buf_key_name : str, ot : str = None, max_counter : int = 100000):
        """
        Initializes a BufferOfDictLists object.

        Parameters
        ----------
        buf_size : int
            Maximum number of buffered items.
        cp : CachePolicy
            Cache policy used to replace buffer items when full.
        buf_name : str
            Name of buffer.
        buf_key_name : str
            Name of items used as buffer keys, e.g. object IDs.
        ot : str, default=None
            Object type to which all buffer items belong; if None, buffer items of any object type are permitted.
        max_counter : int, default=100000
            Maximum counter value (e.g. for key frequency or cache age) before reset.
        """
        self.buf_size = buf_size
        self.cp = cp
        self.buf_name = buf_name
        self.buf_key_name = buf_key_name
        # Used by DFG buffers
        self.ot = ot
        self.max_counter = max_counter

        self.buf_cache_age = 0
        self.buf = dict()
    
    def reduce_to_oids_ots(self, oids_ots_keep : set[str], nested_key : str = None, key_is_tuple : bool = True) -> None:
        """
        Remove buffered items not related to the specified set of object IDs or object types.

        Parameters
        ----------
        oid_ots_keep : set[str]
            Set of object IDs or object types to keep.
        nested_key : str, default=None
            Specifies value-dictionary key to access object ID or object type of a buffer item, if necessary.
        key_is_tuple : bool, default=True
            Specifies if key for accessing object type or object ID is a tuple, e.g. a pair of object types.

        Returns
        -------
        None
        """
        if nested_key is None and not key_is_tuple:
            for buf_key in list(self.buf.keys()):
                if buf_key not in oids_ots_keep:
                    self.remove_item(buf_key)
        elif nested_key is None and key_is_tuple:
            for buf_key_a, buf_key_b in list(self.buf.keys()):
                if buf_key_a is not None and buf_key_a not in oids_ots_keep or buf_key_b is not None and buf_key_b not in oids_ots_keep:
                    self.remove_item((buf_key_a, buf_key_b))
        else:
            for buf_key, buf_dict_list in list(self.buf.items()):
                for i, buf_dict in enumerate(buf_dict_list):
                    if buf_dict[nested_key] not in oids_ots_keep:
                        self.remove_item(buf_key, i)
        
    def get_normalized_rank_by_cp(self) -> dict[Tuple[Any, int], float]:
        """
        Create ranking of individual buffer items based on cache policy of model buffer.
        A buffer item is uniquely defined by its key in the buffer dictionary and the position of its value dictionary in the associated list.

        Returns
        -------
        dict[Tuple[Any, int], float]
            Dictionary mapping each buffered item to a min-max-normalized rank where lowest rank indicates item most likely to be removed according to cache policy.
        
        Raises
        ------
        NotImplementedError
            If cache policy of model buffer is unknown.
        """
        key_to_val = dict()

        if self.cp in [CachePolicy.FIFO, CachePolicy.LRU]:
            counter = 0
            for buf_key, dict_list in self.buf.items():
                for i in range(len(dict_list)):
                    key_to_val[(buf_key, i)] = counter
                    counter += 1
        
        elif self.cp == CachePolicy.LFU:
            key_to_val = self.get_min_max_normalized_key_value_pairs(FREQUENCY)
        
        elif self.cp == CachePolicy.LFU_DA:
            key_to_freq = self.get_min_max_normalized_key_value_pairs(FREQUENCY)
            key_to_cache_age = self.get_min_max_normalized_key_value_pairs(CACHE_AGE)
            
            for buf_key_id in key_to_cache_age:
                key_to_val[buf_key_id] = key_to_cache_age[buf_key_id] + key_to_freq[buf_key_id]
        
        elif self.cp == CachePolicy.RR:
            key_id_pairs = list()
            for buf_key, dict_list in self.buf.items():
                for i in range(len(dict_list)):
                    key_id_pairs.append((buf_key, i))
            
            # Shuffle buffer-key-index pairs to create random ranking
            random.shuffle(key_id_pairs)
            for i in range(len(key_id_pairs)):
                key_to_val[key_id_pairs[i]] = i
        
        else:
            raise NotImplementedError(f'Creating ranking for buffered items not implemented for given cache policy: {self.cp}')

        return min_max_normalize_dict(key_to_val)
    
    def get_min_max_normalized_key_value_pairs(self, key_in_dict : Any) -> dict[Tuple[Any, int], float]:
        """
        Extract pairs of buffer-items keys and min-max-normalized values for a given value-dictionary key.

        Parameters
        ----------
        key_in_dict : Any
            Value-dictionary key for which to extract values for all buffer items.

        Returns
        -------
        dict[Tuple[Any, int], float]
            Dictionary mapping each buffer-item key-index pair to a single, min-max-normalized value.
        """
        key_to_val = dict()

        for buf_key, dict_list in self.buf.items():
            if key_in_dict == FREQUENCY:
                val = len(dict_list)
            
            for i, buffered_dict in enumerate(dict_list):
                if key_in_dict != FREQUENCY:
                    val = buffered_dict[key_in_dict]
                
                key_to_val[(buf_key, i)] = val
        
        return min_max_normalize_dict(key_to_val)

    def get_removal_key_by_cp(self) -> Tuple[Any, int]:
        """
        Determine key and list index of buffer item to remove from full model buffer according to cache policy.

        Returns
        -------
        Tuple[Any, int]
            Key-index pair of buffer item to remove according to cache policy.
        
        Raises
        ------
        NotImplementedError
            If cache policy of model buffer is unknown.
        """
        if self.cp in [CachePolicy.FIFO, CachePolicy.LRU]:
            key_pop = next(iter(self.buf))
            id_pop = 0
        
        elif self.cp == CachePolicy.LFU:
            key_pop = min(self.buf.items(), key=lambda buf_tup: len(buf_tup[1]))[0]
            id_pop = 0
        
        elif self.cp == CachePolicy.LFU_DA:
            key_to_freq = self.get_min_max_normalized_key_value_pairs(FREQUENCY)
            key_to_cache_age = self.get_min_max_normalized_key_value_pairs(CACHE_AGE)
            
            # Translate ranking of buffer key-id pairs according to frequency and cache age into value per key
            key_id_to_rank = dict()
            for buf_key_id in key_to_cache_age:
                key_id_to_rank[buf_key_id] = key_to_cache_age[buf_key_id] + key_to_freq[buf_key_id]

            # The lower the frequency and the lower the cache age, the more likely to be removed
            key_pop, id_pop = min(key_id_to_rank.items(), key=lambda tup: tup[1])[0]

            # Increase cache age whenever there is a removal
            self.buf_cache_age += 1

            # Reset cache ages in buffer once max. counter is reached
            if self.buf_cache_age >= self.max_counter:
                for buffered_key in self.buf:
                    for i in range(len(self.buf[buffered_key])):
                        self.buf[buffered_key][i][CACHE_AGE] = 0
                self.buf_cache_age = 0
        
        elif self.cp == CachePolicy.RR:
            key_id_pop = random.randint(0, len(self.buf)-1)
            key_pop = list(self.buf.keys())[key_id_pop]
            id_pop = random.randint(0, len(self.buf[key_pop])-1)

        else:
            raise NotImplementedError(f'Buffer-item removal not implemented for given cache policy: {self.cp}')
        
        return key_pop, id_pop

    def remove_item(self, key_pop : Any, id_pop : Any = None) -> None:
        """
        Removes buffer item with given key and list index from `buf` dictionary.

        Parameters
        ----------
        key_pop : Any
            Key of buffered item to remove.
        id_pop : Any
            Index of buffered item to remove in list of value dictionaries associated with `key_pop`.

        Returns
        -------
        None
        """
        if id_pop is not None:
            self.buf[key_pop].pop(id_pop)
            # Remove buffer key if associated list of dicts is empty
            if len(self.buf[key_pop]) == 0:
                self.buf.pop(key_pop)
        else:
            self.buf.pop(key_pop)

    def insert_new_item_by_cp(self, new_key : Any, new_dict_overwrite : dict[str, Any]) -> None:
        """
        Adds given buffer item to buffer and replaces existing item according to cache policy if buffer is full.

        Parameters
        ----------
        new_key : Any
            Key of new buffer item, e.g. arc.
        new_dict_overwrite : dict[str, Any]
            Key-value pairs of associated value dictionary that is inserted for new buffer item.

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
            If cache policy of model buffer is unknown.
        """
        if self.cp in [CachePolicy.FIFO, CachePolicy.LFU, CachePolicy.RR]:
            if new_key not in self.buf:
                self.buf[new_key] = list()
            self.buf[new_key].append(new_dict_overwrite)

        elif self.cp == CachePolicy.LFU_DA:
            # Cache age for LFU-DA can only be determined at time of insertion; always needs to be set since each added item creates a new buffer item
            new_dict_overwrite[CACHE_AGE] = self.buf_cache_age
            
            if new_key not in self.buf:
                self.buf[new_key] = list()
            self.buf[new_key].append(new_dict_overwrite)

        elif self.cp == CachePolicy.LRU:
            # Re-append already-seen buffer key and associated list of dicts to maintain LRU order via buffer keys
            existing_dict_list = list()
            if new_key in self.buf:
                existing_dict_list = self.buf[new_key]
                self.buf.pop(new_key)
            self.buf[new_key] = existing_dict_list + [new_dict_overwrite]
        
        else: 
            raise NotImplementedError(f'Adding new buffer item based not implemented for given cache policy: {self.cp}')
    
    def get_buffered_nested_values_for_key(self, nested_key : str) -> set[Any]:
        """
        Returns set of values for given key in buffered value dictionaries.

        Parameters
        ----------
        nested_key : str
            Key in value dictionary for which unique values are extracted.
        
        Returns
        -------
        set[Any]
            Set of values for given nested key.
        """
        buffered_vals = [buf_dict[nested_key] for buf_dict in itertools.chain.from_iterable(list(self.buf.values()))]
        return set(buffered_vals)

    def is_full(self) -> bool:
        """
        Checks if model buffer is full.

        Returns
        -------
        bool
            True if buffer is full, otherwise False.
        """
        return len(self) == self.buf_size

    def __len__(self) -> int:
        """
        Returns length of buffer in terms of the number of buffered items.

        Returns
        -------
        int
            Number of buffered items.
        """
        return len(list(itertools.chain.from_iterable(self.buf.values())))

    def to_string(self, reorder_cols : list[str] = None) -> str:
        """
        Creates string describing buffer including its parameters (cache policy etc.) and buffered items in tabular format.

        Parameters
        ----------
        reorder_cols : list[str], default=None
            List of column names referring to names of buffer-item keys and value-dictionary keys that force order of columns in tabular output.
        
        Returns
        -------
        str
            Output string describing parameters and content of buffer.
        """
        ret_str = f'{self.cp.value} {self.buf_name} characteristics:\n - buffer size: {self.buf_size}\n - max counter: {self.max_counter}\n - object type: {self.ot if self.ot else "--"}\n'

        # Additional transformation for buffers to turn key->list[dict] items into "unrolled" DataFrame rows
        unrolled_buf = {self.buf_key_name: list()} if not reorder_cols else {col_name: list() for col_name in reorder_cols}
        for key, val in self.buf.items():
            if isinstance(val, list):
                for val_dict in val:
                    unrolled_buf[self.buf_key_name].append(key)
                    
                    # Additionally add total frequency of each unrolled e.g. EC-buffer item for cache policies where it matters
                    if self.cp in [CachePolicy.LFU, CachePolicy.LFU_DA]:
                        if FREQUENCY not in unrolled_buf:
                            unrolled_buf[FREQUENCY] = list()
                        unrolled_buf[FREQUENCY].append(len(val))

                    for val_dict_key, val_dict_val in val_dict.items():
                        if val_dict_key not in unrolled_buf:
                            unrolled_buf[val_dict_key] = list()
                        unrolled_buf[val_dict_key].append(val_dict_val)
            
            else:
                raise NotImplementedError(f'Buffer unrolling for tabular pretty-printing not supported! Got unexpected {type(val)} as data type associated w/ buffer keys.')
    
        buf_df = pd.DataFrame.from_dict(unrolled_buf)
        if reorder_cols is not None:
            buf_df = buf_df[reorder_cols]
        
        # Source: https://stackoverflow.com/questions/18528533/pretty-printing-a-pandas-dataframe
        ret_str += tabulate(buf_df, headers='keys', tablefmt='psql', showindex=True)
        ret_str += '\n'

        return ret_str