import pprint
import argparse
import pandas as pd
import plotly.express as px
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
    common_gaps = all_available_times[all_available_times['Duration'] > interval].copy()

    common_gaps['Duration'] = common_gaps['Duration'] - interval
    common_gaps['Timestamp'] = common_gaps['Timestamp'] - common_gaps['Duration']

    common_gaps = common_gaps.reset_index(drop=True)

    return common_gaps


def prepare_dataframe(df: pd.DataFrame, common_gaps_df: pd.DataFrame, interval: timedelta):
    all_available_times = df['Timestamp'].unique()
    all_available_times.sort()
    frequency = str(interval.seconds) + 'S'
    prototype_df = pd.DataFrame(pd.date_range(start=all_available_times[0], end=all_available_times[-1], freq=frequency), columns=('Timestamp',))
    prototype_df['Status'] = 'None'

    for index in common_gaps_df.index:
        gap_start = common_gaps_df.loc[index]['Timestamp']
        gap_end = common_gaps_df.loc[index]['Timestamp'] + common_gaps_df.loc[index]['Duration']
        gaps_df = prototype_df[prototype_df['Timestamp'] >= gap_start]
        gaps_df = gaps_df[gaps_df['Timestamp'] < gap_end]
        prototype_df.loc[gaps_df.index, 'Status'] = 'Common gap'


    ret_df = pd.DataFrame()

    sats = df['Satellite'].unique()
    sats.sort()
    for sat in sats:
        sat_df = df[df['Satellite'] == sat]
        prototype_df_copy = prototype_df.copy()
        prototype_df_copy['Satellite'] = sat
        prototype_df_copy['isin'] = prototype_df_copy['Timestamp'].isin(sat_df['Timestamp'])
        prototype_df_copy.loc[prototype_df_copy[prototype_df_copy['isin'] == True].index, 'Status'] = 'Data'
        prototype_df_copy = prototype_df_copy.drop(columns=['isin',])
        ret_df = pd.concat([ret_df, prototype_df_copy], ignore_index=True)

    return ret_df


def check_density_of_gaps(df: pd.DataFrame, window_size: str, max_gap_num: int):
    windows = []
    i = 0

    start_time = None

    for window in df.rolling(window=window_size, on='Timestamp'):
        if len(window) == (len(window[window['Status'] == 'None']) + len(window[window['Status'] == 'Common gap'])):
            continue

        if len(window) > 1:
            check_left = window.iloc[0 : len(window) // 2]
            check_right = window.iloc[len(window) // 2 : len(window)]

            if len(check_left[check_left['Status'] == 'Data']) and len(check_right[check_right['Status'] == 'None']) >= len(window) // 2:
                if len(windows) > i:
                    windows[i].difference_update(window.index.to_pydatetime())
                continue

            if window.iloc[-1]['Status'] == 'Data' and len(window[window['Status'] == 'Data']) == 1:
                start_time = window.iloc[-1].name
        
        gaps = window[window['Status'] == 'None']
        if len(gaps) > max_gap_num:
            if not len(windows) > 0:
                windows.append(set())
            if len(windows[i]) == 0 or gaps.index[0] in windows[i]:
                windows[i].update(gaps.index.to_pydatetime())
                if start_time:
                    for val in list(windows[i]):
                        if val < start_time:
                            windows[i].discard(val)
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
    args = parser.parse_args()

    interval = timedelta(seconds=args.interval)
    window_size = str(args.window_size) + 'S'

    df = pd.DataFrame()

    for file in args.files:
        print(f'Read {file}')
        temp_df = read_to_df(file)
        df = pd.concat([df, temp_df], ignore_index=True)

    common_gaps_df = find_common_gaps(df, interval)

    working_df = prepare_dataframe(df, common_gaps_df, interval)

    problems_by_sat = {}
    for sat in working_df['Satellite'].unique():
        problems_by_sat[sat] = check_density_of_gaps(working_df[working_df['Satellite'] == sat], window_size, args.max_gap_num)

    if args.plot_show or not args.plot_file == None:
        create_debug_plot(working_df, problems_by_sat, interval, show=args.plot_show, filename=args.plot_file)

    # print('Common problems')
    # pprint.pprint(common_problems)
    # print('Problems by satellite')
    # pprint.pprint(problems_by_sat)
