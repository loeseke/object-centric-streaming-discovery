"""\
Script for cleaning up OCEL 2.0 ContainerLogistics log in XML format so that it matches corresponding JSON log.
Ensure that the downloaded XML log is named "ContainerLogistics_original.xml". Corrected output is written to "ContainerLogistics.xml".
Usage: python fix_xml_ContainerLogistics.py
__author__: "Nina Löseke"
"""

import xml.etree.ElementTree as ET
from datetime import datetime
import pytz

if __name__ == '__main__':
    # Convert timestamps for ContainerLogistics XML to Python datetime-readable format
    tree = ET.parse('./ContainerLogistics_original.xml')
    root = tree.getroot()

    # Timestamps of object attributes have format "Mon May 22 2023 13:54:42 GMT+0200 (Mitteleuropäische Sommerzeit)", sometimes "Normalzeit"
    date_format_objects = '%a %b %d %Y %H:%M:%S %Z%z'

    for attr in root.findall('objects/object/attributes/attribute'):
        # Replace "null" object attribute values w/ "-1" (not otherwise used in log as attribute value) since ocpa ocel2 importer expects float
        text_str = attr.text
        if text_str == 'null':
            attr.text = '-1'

        time_str = attr.get('time')
        time_str = time_str.split(' (')[0]
        time_dt = datetime.strptime(time_str, date_format_objects).astimezone(pytz.utc)
        attr.set('time', time_dt.isoformat())
    
    # Timestamps of events have format "2024-05-14T12:09:46.000Z"
    date_format_events = '%Y-%m-%dT%H:%M:%S.%fZ'

    for attr in root.findall('events/event'):
        time_str = attr.get('time')
        time_dt = datetime.strptime(time_str, date_format_events)
        attr.set('time', time_dt.isoformat())
    
    # Remove E2O objects that have no defined types (done automatically for EventStream)
    oid_to_ot = dict()

    for obj in root.findall('objects/object'):
        oid = obj.get('id')
        ot = obj.get('type')
        oid_to_ot[oid] = ot
    
    for obj in root.findall('events/event/objects'):
        for e2o in obj.findall('relationship'):
            oid = e2o.get('object-id')
            if oid not in oid_to_ot:
                obj.remove(e2o)

    # Write corrected version back to file
    tree.write('ContainerLogistics.xml')