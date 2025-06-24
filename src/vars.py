"""\
Fixed variables used throughout framework.
__author__: "Nina Löseke"
"""

# Define keys of dictionaries associated w/ buffer keys to avoid mistakes
LAST_SEEN = 'last seen'
FIRST_SEEN = 'first seen'
ACTIVITY = 'activity'
FREQUENCY = 'frequency'
CACHE_AGE = 'cache age'
TARGET_ACTIVITY_FREQ = 'target-activity freq'
OID = 'oid'
ACTIVITY_DURATION = 'activity duration'
OBJECT_TYPE = 'object type'
STRIDES = 'strides'

EVENT_CARD = 'event cardinalities'
EVENT_ID = 'unique event ID'
OBJECT_PAIR = 'object pair'

HAS_SINGLE_OBJ = 'has single object'

# Define keys for dictionaries for OC-DFG & OCPN discovery
NODE_BUF_KEY = 'oid'
ARC_BUF_KEY = 'arc'

# Define keys of dictionaries for TOTeM discovery
TR_BUF_KEY = 'oid'
TR_DURING = "during"
TR_DURING_INVERSE = "during_inv"
TR_PRECEDES = "precedes"
TR_PRECEDES_INVERSE = "precedes_inv"
TR_PARALLEL = "parallel"

EC_BUF_KEY = 'directed OT pair'
EC_ZERO = "0"
EC_ONE = "1"
EC_ZERO_ONE = "0..1"
EC_ONE_MANY = "1..*"
EC_ZERO_MANY = "0..*"

LC_BUF_KEY = 'undirected OT pair'
LC_ONE = "1"
LC_ZERO_ONE = "0..1"
LC_ONE_MANY = "1..*"
LC_ZERO_MANY = "0..*"

# Define keys for dictionaries for OCPN discovery
EA_BUF_KEY = 'activity'

# Define column names for Monitor DataFrames
M_STREAM_ITEM = 'stream item'
M_CACHE_HITS = '# hits'
M_CACHE_MISSES = '# misses'
M_BUF_NAME = 'buffer name'
M_OBJECT_TYPE = 'object type'
M_OT_PERCENTAGE = 'OT percentage'
M_PPB_SIZE = 'priority-policy buffer size'
M_BUF_UPDATE_TIME = 'buffer-update time'
M_ITEM_PROCESSING_TIME = 'stream-item processing time'
M_ITEM_TYPE = 'stream-item type'

# Global variables for graphviz
GV_FONT = 'DejaVu Sans'     # match matplotlib default
GV_NODE_FONTSIZE = '10pt'
GV_EDGE_FONTSIZE = '8pt'
GV_GRAPH_FONTSIZE = '8pt'