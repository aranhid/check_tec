import os
import argparse
import pandas as pd
from math import isclose
from gnss_tec import rnx
from datetime import datetime, timedelta


def find_gaps(file: str, deltatime: timedelta):
    all_available_times = set()
    gaps_by_sat = {}
    with open(file) as obs_file:
        reader = rnx(obs_file)
        prev_timestamps = {}
        for observables in reader:
            current_timestamp = observables.timestamp
            current_satellite = observables.satellite

            all_available_times.add(current_timestamp)

            if not current_satellite in prev_timestamps.keys():
                prev_timestamps[current_satellite] = current_timestamp
            else:
                if not current_satellite in gaps_by_sat.keys():
                        gaps_by_sat[current_satellite] = []
                gap_duration = current_timestamp - prev_timestamps[current_satellite]
                if gap_duration > deltatime:
                    gaps_by_sat[current_satellite].append((current_timestamp - gap_duration + deltatime, gap_duration - deltatime))
                prev_timestamps[current_satellite] = current_timestamp

    all_available_times = sorted(all_available_times)
    df = pd.DataFrame(list(all_available_times), columns=('Timedelta',))
    timedeltas = df.diff()
    all_sats_gaps = timedeltas[timedeltas['Timedelta'] > deltatime]
    gaps = []
    for index in all_sats_gaps.index:
        gap_time = df.loc[index]['Timedelta'].to_pydatetime() - gap_duration + deltatime
        gap_duration = timedeltas.loc[index]['Timedelta'].to_pytimedelta() - deltatime
        gaps.append((gap_time, gap_duration))

    filtered_gaps_by_sat = {}

    for sat in gaps_by_sat.keys():
        filtered_gaps = [gap for gap in gaps_by_sat[sat] if gap not in gaps]
        if len(filtered_gaps) > 0:
            filtered_gaps_by_sat[sat] = filtered_gaps

    gaps_df = pd.DataFrame(data=[gap[1] for gap in gaps], index=[
                           gap[0] for gap in gaps], columns=('Timedelta', ))
    gaps_df = gaps_df.sort_index()

    gaps_by_sat_df = {}
    for sat in gaps_by_sat.keys():
        temp_df = pd.DataFrame(data=[gap[1] for gap in gaps_by_sat[sat]], index=[
                                           gap[0] for gap in gaps_by_sat[sat]], columns=('Timedelta', ))
        temp_df = temp_df.sort_index()
        gaps_by_sat_df[sat] = temp_df

    return gaps_df, gaps_by_sat_df


def check_duration_of_gaps(df: pd.DataFrame, deltatime: timedelta):
    max_timedelta = df[df['Timedelta'] > deltatime]
    print('Duration')
    print(max_timedelta)
    # if len(max_timedelta) > 0:
    #     print('Не прошло по длине пропуска')
    #     return
    # print('Прошло по длине пропуска')


def check_density_of_gaps(df: pd.DataFrame, window_size: str, max_gap_count: int):
    windows = []
    windows.append(set())
    i = 0

    for window in df.rolling(window_size):
        print(window)
        if len(window) > max_gap_count:
            if len(windows[i]) == 0:
                for index in window.index:
                    windows[i].add(index)
            elif window.index[0] in windows[i]:
                for index in window.index:
                    windows[i].add(index)
            else:
                windows.append(set())
                i += 1
                for index in window.index:
                    windows[i].add(index)

    windows = [sorted(window) for window in windows]
    
    ret = []
    for window in windows:
        if len(window):
            ret.append((window[0], window[-1]))

    print(ret)

    return ret


def main_f(file, interval, window_size, max_gap_num):
    common_gaps_df, gaps_by_sat_df = find_gaps(file, interval)

    problems_by_sat = {}

    for sat in gaps_by_sat_df.keys():
        problems_by_sat[sat] = check_density_of_gaps(gaps_by_sat_df[sat], window_size, max_gap_num)

    for sat in problems_by_sat.keys():
        print(sat)
        for problem in problems_by_sat[sat]:
            print(problem)
        



if __name__ == '__main__':
    file1 = 'observables\IST5063W.22O' # одна дыра
    file2 = 'C:\\Users\\vladm\\Documents\\Работа\\chart +\\observables\\IST5062F.22O' # несколько дыр 
    file3 = "C:\\Users\\vladm\\Documents\\Работа\\chart +\\march9\\SEPT0680.22O" # большой файл (не проходит проверки)
    file4 = "C:\\Users\\vladm\\Documents\\Работа\\chart +\\100H\\IST5069H.22O" # после 100 герц, очень много пропусков
    march11_13 = "R:\\Septentrio\\22070\\IST5070F.22O"
    march11_15 = "R:\\Septentrio\\22070\\IST5070H.22O"
    march11_15_20msec = "C:\\Users\\vladm\\Documents\\Работа\\chart +\\march11\\SEPT0700.22O"
    # interval = timedelta(milliseconds=20)
    interval = timedelta(seconds=30)
    window_size = '10min'
    max_gap_num = 1
    main_f(march11_13, interval, window_size, max_gap_num)

    # parser = argparse.ArgumentParser()
    # parser.add_argument('file', type=str, help='path to RINEX file')
    # parser.add_argument('interval', type=int, help='interval of RINEX file, in seconds')
    # parser.add_argument('window_size', type=int, help='size for the rolling window to check gaps, in seconds')
    # parser.add_argument('max_gap_num', type=int, help='maximum number of gaps in the rolling window')
    # args = parser.parse_args()

    # main_f(args.file, args.interval, str(args.window_size) + 'S', args.gap_num)