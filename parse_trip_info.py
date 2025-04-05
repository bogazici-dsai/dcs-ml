'''
@Author: pangay 1623253042@qq.com
@Date: 2024-03-27 14:54:18
@LastEditors: pangay 1623253042@qq.com
@LastEditTime: 2024-06-26 01:08:30
@FilePath: /TSC-HARLA/parse_trip_info.py
@Description: Parse SUMO tripinfo XML file and export traffic statistics as CSV.
'''

from tshub.utils.parse_trip_info import TripInfoStats
from tshub.utils.get_abs_path import get_abs_path

# Resolve path helper
path_convert = get_abs_path(__file__)

if __name__ == '__main__':
    # Define the path to the SUMO-generated tripinfo XML file
    trip_info_path = path_convert('./Result/rule.tripinfo.xml')

    # Parse and compute trip statistics
    stats = TripInfoStats(trip_info_path)

    # Save parsed data as CSV
    stats.to_csv(path_convert('./Result/rule.csv'))