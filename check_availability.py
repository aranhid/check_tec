import pprint
import argparse
import pandas as pd
from math import isclose
from types import MappingProxyType
from gnss_tec import rnx, BAND_PRIORITY
from datetime import datetime, timedelta


def read_to_df(file: str, band_priority: MappingProxyType = BAND_PRIORITY):
    data = []

    with open(file) as obs_file:
        reader = rnx(obs_file, band_priority=band_priority)
        for observables in reader:
            data.append((
                observables.satellite,
                observables.timestamp,
            ))

    df = pd.DataFrame(data, columns=("Satellite", "Timestamp"))

    return df


def find_common_gaps(df: pd.DataFrame, interval: timedelta):
    all_available_times = df['Timestamp'].unique()
    all_available_times = sorted(all_available_times)
    all_available_times = pd.DataFrame(list(all_available_times), columns=('Timestamp',))
    all_available_times['Duration'] = all_available_times.diff()
    all_sats_gaps = all_available_times[all_available_times['Duration'] > interval].copy()

    all_sats_gaps['Duration'] = all_sats_gaps['Duration'] - interval
    all_sats_gaps['Timestamp'] = all_sats_gaps['Timestamp'] - all_sats_gaps['Duration']

    all_sats_gaps = all_sats_gaps.reset_index(drop=True)

    return all_sats_gaps


def find_gaps_in_sats(df: pd.DataFrame, interval: timedelta, common_gaps: pd.DataFrame):
    sats = df['Satellite'].unique()

    gaps_by_sat = {}

    for sat in sats:
        sat_df = df[df['Satellite'] == sat]
        sat_df = sat_df.sort_values(by=['Timestamp'])
        sat_df['Duration'] = sat_df['Timestamp'].diff()
        gaps = sat_df[sat_df['Duration'] > interval]
        gaps = gaps.drop(columns=['Satellite',])
        gaps['Duration'] = gaps['Duration'] - interval
        gaps['Timestamp'] = gaps['Timestamp'] - gaps['Duration']
        gaps = gaps.reset_index(drop=True)


        merge_df = gaps.merge(common_gaps, on=['Timestamp','Duration'], 
                   how='left', indicator=True)

        gaps_without_common_gaps = merge_df[merge_df['_merge'] == 'left_only']
        gaps_without_common_gaps = gaps_without_common_gaps.drop(columns=['_merge',])
        gaps_by_sat[sat] = gaps_without_common_gaps

    return gaps_by_sat


def check_density_of_gaps(df: pd.DataFrame, window_size: str, max_gap_num: int):
    work_df = df.copy()
    work_df = work_df.set_index(keys='Timestamp', drop=True)
    windows = []
    i = 0

    if len(work_df):
        for window in work_df.rolling(window_size):
            if len(window) > max_gap_num:
                if not len(windows) > 0:
                    windows.append(set())
                if len(windows[i]) == 0 or (window.index[0], window.loc[window.index[0]]['Duration']) in windows[i]:
                    for index in window.index:
                        windows[i].add((index, window.loc[index]['Duration']))
                else:
                    windows.append(set())
                    i += 1
                    for index in window.index:
                        windows[i].add((index, window.loc[index]['Duration']))

    windows = [sorted(window) for window in windows]
    
    ret = []
    for window in windows:
        if len(window):
            ret.append((window[0][0], window[-1][0] + window[-1][1]))

    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='path to RINEX file')
    parser.add_argument('interval', type=float, help='interval of RINEX file, in seconds')
    parser.add_argument('window_size', type=float, help='size for the rolling window to check gaps, in seconds')
    parser.add_argument('max_gap_num', type=int, help='maximum number of gaps in the rolling window')
    args = parser.parse_args()

    interval = timedelta(seconds=args.interval)
    window_size = str(args.window_size) + 'S'

    df = read_to_df(args.file)

    common_gaps_df = find_common_gaps(df, interval)
    common_problems = check_density_of_gaps(common_gaps_df, window_size, args.max_gap_num)

    gaps_by_sat_df = find_gaps_in_sats(df, interval, common_gaps_df)
    problems_by_sat = {}
    for sat in gaps_by_sat_df.keys():
        problems_by_sat[sat] = check_density_of_gaps(gaps_by_sat_df[sat], window_size, args.max_gap_num)
    

    print('Common problems')
    pprint.pprint(common_problems)
    print('Problems by satellite')
    pprint.pprint(problems_by_sat)