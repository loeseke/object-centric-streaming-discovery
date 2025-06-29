"""\
Script for automatically downloading example OCEL 2.0 logs for testing streaming framework.
Usage: python download_logs.py
__author__: "Nina Löseke"
"""

import requests
import os
from zipfile import PyZipFile

# Example OCEL 2.0 logs to download to data directory
URLS = {
    'ContainerLogistics.json': 'https://zenodo.org/records/8428084/files/ContainerLogistics.json',
    'ContainerLogistics_original.xml': 'https://zenodo.org/records/8428084/files/ContainerLogistics.xml',
    'AgeOfEmpires.zip': 'https://zenodo.org/records/13365584/files/age_of_empires_ocel2_10_match_filter.zip'
}

if __name__ == "__main__":
    # Source: https://www.geeksforgeeks.org/how-to-download-files-from-urls-with-python/
    for output_file, url in URLS.items():
        if not os.path.isfile(output_file):
            response = requests.get(url)
            if response.status_code == 200:
                with open(output_file, 'wb') as file:
                    file.write(response.content)
                print(f'{output_file} downloaded!')

    # Download reduced version of Age of Empires log with only 10 matches
    zip_folder = PyZipFile('./AgeOfEmpires.zip')
    zip_folder.extract('age_of_empires_ocel2_10_match_filter.json')
    zip_folder.extract('age_of_empires_ocel2_10_match_filter.xml')
    zip_folder.close()
    os.rename('age_of_empires_ocel2_10_match_filter.json', 'AgeOfEmpires10Matches.json')
    os.rename('age_of_empires_ocel2_10_match_filter.xml', 'AgeOfEmpires10Matches.xml')
    os.remove('AgeOfEmpires.zip')