import pprint
import argparse
import pandas as pd
import plotly.express as px
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
        gaps_without_common_gaps = gaps_without_common_gaps.reset_index(drop=True)
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


def create_simple_plot(df: pd.DataFrame, interval: timedelta):
    work_df = df.copy()
    work_df['Timestamp end'] = work_df['Timestamp'] + interval
    fig = px.timeline(work_df, x_start="Timestamp",
                      x_end="Timestamp end", y="Satellite")
    fig.show()


def create_debug_plot(df: pd.DataFrame, interval: timedelta, common_gaps: pd.DataFrame, gaps_by_sat, problems_by_sat):
    work_df = df.copy()
    work_df = work_df.sort_values(by=('Satellite'))
    work_df['Status'] = 'Normal'
    work_df['Timestamp end'] = work_df['Timestamp'] + interval

    for sat in work_df['Satellite'].unique():
        common_gaps_copy = common_gaps.copy()
        common_gaps_copy['Status'] = 'Common gap'
        common_gaps_copy['Satellite'] = sat
        common_gaps_copy['Timestamp end'] = common_gaps_copy['Timestamp'] + common_gaps_copy['Duration']
        work_df = pd.concat([work_df, common_gaps_copy])

    for sat in gaps_by_sat_df.keys():
        sat_gaps_copy = gaps_by_sat[sat]
        sat_gaps_copy['Status'] = 'Satellite gap'
        sat_gaps_copy['Satellite'] = sat
        sat_gaps_copy['Timestamp end'] = sat_gaps_copy['Timestamp'] + sat_gaps_copy['Duration']
        work_df = pd.concat([work_df, sat_gaps_copy])

    work_df = work_df.reset_index(drop=True)

    for sat in problems_by_sat:
        if len(problems_by_sat[sat]) > 0:
            for problem in problems_by_sat[sat]:
                sat_df = work_df[work_df['Satellite'] == sat]
                sat_df = sat_df[sat_df['Timestamp'] >= problem[0]]
                sat_df = sat_df[sat_df['Timestamp'] < problem[1]]
                for index in sat_df.index:
                    work_df.loc[index, ('Status',)] = 'Problem window'

    discrete_map_resource = { 'Problem window': '#FF0000', 'Common gap': '#00FF00', 'Normal': '#0000FF', 'Satellite gap': '#FFFF00'}
    fig = px.timeline(work_df, x_start='Timestamp',
                      x_end='Timestamp end', y='Satellite', color='Status', color_discrete_map=discrete_map_resource)
    fig.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', type=str, nargs='+', help='path to RINEX file')
    parser.add_argument('--interval', type=float, help='interval of RINEX file, in seconds')
    parser.add_argument('--window-size', type=float, help='size for the rolling window to check gaps, in seconds')
    parser.add_argument('--max-gap-num', type=int, help='maximum number of gaps in the rolling window')
    args = parser.parse_args()

    interval = timedelta(seconds=args.interval)
    window_size = str(args.window_size) + 'S'

    df = pd.DataFrame()

    for file in args.files:
        print(f'Read {file}')
        temp_df = read_to_df(file)
        df = pd.concat([df, temp_df], ignore_index=True)

    common_gaps_df = find_common_gaps(df, interval)
    common_problems = check_density_of_gaps(common_gaps_df, window_size, args.max_gap_num)

    gaps_by_sat_df = find_gaps_in_sats(df, interval, common_gaps_df)
    problems_by_sat = {}
    for sat in gaps_by_sat_df.keys():
        problems_by_sat[sat] = check_density_of_gaps(gaps_by_sat_df[sat], window_size, args.max_gap_num)
    
    create_debug_plot(df, interval, common_gaps_df, gaps_by_sat_df, problems_by_sat)

    print('Common problems')
    pprint.pprint(common_problems)
    print('Problems by satellite')
    pprint.pprint(problems_by_sat)