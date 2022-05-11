import pprint
import argparse
import pandas as pd
import plotly.express as px
from datetime import timedelta

from reader import get_dataframe


def find_common_problems(df: pd.DataFrame, interval: timedelta):
    sats = df['Satellite'].unique()
    gaps = df[df['Status'] == 'Common gap']
    gaps = gaps[gaps['Satellite'] == sats[0]].copy()

    gap_time_start = []
    gap_time_start.append(gaps.iloc[0]['Timestamp'].to_pydatetime())
    gaps['diff'] = gaps['Timestamp'].diff()
    borders = gaps[gaps['diff'] > interval]
    gap_time_start.extend(borders['Timestamp'].dt.to_pydatetime())
    
    
    gap_time_end = []
    gap_time_end.extend((borders['Timestamp'] - borders['diff'] + interval).dt.to_pydatetime())
    gap_time_end.append(gaps.iloc[-1]['Timestamp'].to_pydatetime())
    
    common_problems = list(zip(gap_time_start, gap_time_end))
    return common_problems


def check_density_of_gaps(df: pd.DataFrame, interval: timedelta, window_size: float, max_gap_num: int):
    window_len = window_size // interval.total_seconds()
    window_size_str = str(window_size) + 'S'
    windows = []
    i = 0

    times_with_elevation = df[df['Elevation'].notna()]
    if len(times_with_elevation):
        for window in times_with_elevation.rolling(window=window_size_str, on='Timestamp'):
            if len(window) < window_len:
                continue
            gaps = window[window['Status'] == 'None']
            if len(gaps) > max_gap_num:
                if not len(windows):
                    windows.append(set())
                if len(windows[i]) == 0 or gaps.index[0] in windows[i]:
                    windows[i].update(gaps.index.to_pydatetime())
                else:
                    windows.append(set())
                    i += 1
                    windows[i].update(gaps.index.to_pydatetime())

    windows = [sorted(window) for window in windows]
    
    ret = []
    for window in windows:
        if len(window):
            ret.append((window[0], window[-1]))

    return ret


def create_simple_plot(df: pd.DataFrame, interval: timedelta):
    work_df = df.copy()
    work_df['Timestamp end'] = work_df['Timestamp'] + interval
    fig = px.timeline(work_df, x_start="Timestamp",
                      x_end="Timestamp end", y="Satellite")
    fig.show()


def create_debug_plot(df: pd.DataFrame, problems_by_sat: dict, interval: timedelta, filename: str = None, show: bool = False):
    work_df = df.copy()
    work_df['Timestamp end'] = work_df['Timestamp'] + interval

    for sat in problems_by_sat.keys():
        for problem in problems_by_sat[sat]:
            problem_df = work_df[work_df['Satellite'] == sat]
            problem_df = problem_df[problem_df['Timestamp'] >= problem[0]]
            problem_df = problem_df[problem_df['Timestamp'] <= problem[1]]
            work_df.loc[problem_df.index, 'Status'] = 'Problem'

    discrete_map_resource = {'Problem': '#FF0000', 'Data': '#00b300', 'Common gap': '#0000FF', 'None': '#d9d507'}
    fig = px.timeline(work_df, x_start='Timestamp', x_end='Timestamp end', y='Satellite', color='Status', color_discrete_map=discrete_map_resource)
    if show:
        fig.show()
    if filename:
        fig.write_image(filename, width=1920, height=1080)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', type=str, nargs='+', help='path to RINEX file')
    parser.add_argument('--interval', type=float, help='interval of RINEX file, in seconds')
    parser.add_argument('--window-size', type=float, help='size for the rolling window to check gaps, in seconds')
    parser.add_argument('--max-gap-num', type=int, help='maximum number of gaps in the rolling window')
    parser.add_argument('--plot-show', action='store_true', help='show plot')
    parser.add_argument('--plot-file', type=str, default=None, help='path for plot image')
    parser.add_argument('--nav-file', type=str, default=None, help='path to NAV file')
    parser.add_argument('--cutoff', type=float, help='Cutoff for elevation')
    args = parser.parse_args()

    interval = timedelta(seconds=args.interval)

    working_df = get_dataframe(args.files, interval, args.nav_file, args.cutoff)
    common_problems = find_common_problems(working_df, interval)
    problems_by_sat = {}
    if args.nav_file:
        print('Find problems by satellite')
        for sat in working_df['Satellite'].unique():
            print(f'Process {sat}')
            problems_by_sat[sat] = check_density_of_gaps(working_df[working_df['Satellite'] == sat], interval, args.window_size, args.max_gap_num)

    print('Create plot')
    if args.plot_show or not args.plot_file == None:
        create_debug_plot(working_df, problems_by_sat, interval, show=args.plot_show, filename=args.plot_file)

    # print('Common problems')
    # pprint.pprint(common_problems)
    # print('Problems by satellite')
    # pprint.pprint(problems_by_sat)
