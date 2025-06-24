# Object-Centric Streaming Discovery

This streaming framework supports the online discovery of Object-Centric Directly-Follows Graphs (OC-DFGs), Object-Centric Petri Nets (OCPNs), and Temporal Object Type Models (TOTeM models) from object-centric event streams. Currently, object-centric event streams are simulated based on OCEL 2.0 logs in JSON format.

## Setup

The implementation is based on Python 3.12. For evaluation purposes, each model can be discovered offline on a full log and translated to the representation used by the streaming framework. The TOTeM offline discovery requires version 1.3.3 of the `ocpa` library, which requires version 2.2.32 of `pm4py`. The offline discovery of OC-DFGs and OCPNs, on the other hand, does not rely on `ocpa` and uses `pm4py` version 2.7.15.
Therefore, two different virtual environments need to be set up to evaluate each online model against its offline model.

```bash
# Set up venv for OC-DFG and OCPN discovery
python -m venv path/to/pm4py/venv
source path/to/pm4py/venv/Scripts/activate
pip install -r requirements_pm4py.txt
```
```bash
# Set up venv for TOTeM discovery
python -m venv path/to/ocpa/venv
source path/to/ocpa/venv/Scripts/activate
pip install -r requirements_ocpa.txt
```

## Usage

The `data` directory contains a toy OCEL 2.0 log and scripts for automatically downloading a large OCEL 2.0 log and pre-processing it. Classes and functions for converting a log into an event stream, processing the stream by updating a streaming representation, and mining, visualizing, and evaluating the resulting model are located in the `src` directory.

### Downloading OCEL 2.0 log

```bash
source path/to/pm4py/venv/Scripts/activate
cd data
python download_logs.py
python fix_xml_ContainerLogistics.py
./format_jsons.sh
```

### Converting log to object-centric event stream

```python
from utils import EventStream

event_stream = EventStream(path/to/json/ocel, o2o_has_time=False)
```

### Processing stream

Example workflow for creating event stream, defining parameters of streaming representation, and processing stream; works analogously for `OcdfgBufferPerObjectType`, `OcpnBuffer`, and `TotemBuffer`:

```python
from utils import EventStream
from model_buffers import OcdfgBuffer, OcdfgBufferPerObjectType, OcpnBuffer, TotemBuffer
from cache_policy_buffers import CachePolicy
from priority_policy_buffers import PPBLifespanPerObject, PrioPolicyOrder

# Create event stream
event_stream = EventStream(path/to/json/ocel, o2o_has_time=False)

# Set up streaming representations with model buffers e.g. of size 100 each and with mandatory cache policy and optional priority policy
ocdfg_buf = OcdfgBuffer(
    100, 
    100, 
    CachePolicy.LRU, 
    pp_buf=PPBLifespanPerObject(prio_order=PrioPolicyOrder.MAX), 
    coupled_removal=False
)

# Process entire stream; can enable/disable that additional O2O relations are derived from in-coming events
ocdfg_buf.process_stream(event_stream.stream, enrich_o2o=False)
```

### Discovering and visualizing model from streaming representation

Assume `ocdfg_buf` and `event_stream` as above. Models are mined and visualized analogously for OCPNs via `OcpnModel` and for TOTeM via `TotemModel`:

```python
[...]
from model_builder_ocdfg import OcdfgModel
from model_builder_ocpn import OcpnModel
from model_builder_totem import TotemModel
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np

# Here, 10% of least frequent nodes and arcs are removed from resulting OC-DFG
ocdfg_model = OcdfgModel(ocdfg_buf, prune_node_frac=0.1, prune_arc_frac=0.1, verbose=False)

# Define color per object type for visualization
ots = sorted(event_stream.object_types)
ot_rgb_colors = cm.jet(np.linspace(0, 1, len(ots)))
ot_to_rgb_color = dict(zip(ots, ot_rgb_colors))
ot_to_hex_color = {ot: mpl.colors.rgb2hex(ot_rgb) for ot, ot_rgb in ot_to_rgb_color.items()}
ocdfg_model.visualize(path/to/output/directory, 'ocdfg.pdf', ot_to_hex_color, visualize_dfgs=True)
```

## Authors

The streaming framework for discovering object-centric models was created by [Nina Löseke](https://github.com/loeseke). [Lukas Liss](https://github.com/LukasLiss) contributed the original code of the [Temporal Object Type Model (TOTeM) Miner](https://github.com/LukasLiss/TOTeM-temporal-object-type-model).