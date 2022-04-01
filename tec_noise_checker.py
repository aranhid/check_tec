import os
import math
import argparse
from pprint import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from types import MappingProxyType
from datetime import datetime, timedelta
from gnss_tec import rnx, gnss, BAND_PRIORITY

from locate_sat import get_elevations


def calculate_combinations(df):
    Phi_LC_list = []
    R_PC_list = []
    Phi_LI_list = []
    R_PI_list = []
    Phi_LW_list = []
    R_PW_list = []
    Phi_LN_list = []
    R_PN_list = []
    for index, row in df.iterrows():
        sat = row["Satellite"][0]
        f1 = gnss.FREQUENCY.get(sat).get(int(row["Phase code"].get(1)[1]))
        f2 = gnss.FREQUENCY.get(sat).get(int(row["Phase code"].get(2)[1]))
        Phi_L1 = row["Phase"].get(1)
        Phi_L2 = row["Phase"].get(2)
        R_P1 = row["P range"].get(1)
        R_P2 = row["P range"].get(2)
        
        # Ionosphere-free combination
        Phi_LC = (f1**2*Phi_L1 - f2**2*Phi_L2)/(f1**2-f2**2)
        R_PC = (f1**2*R_P1 - f2**2*R_P2)/(f1**2-f2**2)

        Phi_LC_list.append(Phi_LC)
        R_PC_list.append(R_PC)

        # Geometry-free combination
        Phi_LI = Phi_L1 - Phi_L2
        R_PI = R_P2 - R_P1

        Phi_LI_list.append(Phi_LI)
        R_PI_list.append(R_PI)

        # Wide-laning combinations
        Phi_LW = (f1*Phi_L1 - f2*Phi_L2)/(f1-f2)
        R_PW = (f1*R_P1 - f2*R_P2)/(f1-f2)

        Phi_LW_list.append(Phi_LW)
        R_PW_list.append(R_PW)

        # Narrow-laning combinations
        Phi_LN = (f1*Phi_L1 - f2*Phi_L2)/(f1-f2)
        R_PN = (f1*R_P1 - f2*R_P2)/(f1-f2)

        Phi_LN_list.append(Phi_LN)
        R_PN_list.append(R_PN)
    
    return df.assign(Phi_LC = Phi_LC_list, 
                     R_PC = R_PC_list,
                     Phi_LI = Phi_LI_list,
                     R_PI = R_PI_list,
                     Phi_LW = Phi_LW_list,
                     R_PW = R_PW_list,
                     Phi_LN = Phi_LN_list,
                     R_PN = R_PN_list)


def read_to_df(file: str, band_priority: MappingProxyType = BAND_PRIORITY):
    data = []

    with open(file) as obs_file:
        reader = rnx(obs_file, band_priority=band_priority)
        for observables in reader:
            sat = observables.satellite
            if sat[1] == ' ':
                sat = sat[0] + '0' + sat[2]
            data.append((
                sat,
                observables.timestamp,
                observables.phase_code,
                observables.phase,
                observables.phase_tec,
                observables.p_range_code,
                observables.p_range,
                observables.p_range_tec,
            ))

    df = pd.DataFrame(data, columns=("Satellite", "Timestamp", "Phase code", "Phase", "Phase tec", "P range code", "P range", "P range tec"))

    return df


def get_windows(data, window, step):
    win = np.arange(window)[None, :]
    shift = np.arange(0, data.shape[0] - win[0][-1], step)[:, None]
    indexer = win + shift
    return indexer


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
    prototype_df = pd.DataFrame(
        pd.date_range(start=all_available_times[0], end=all_available_times[-1], freq=frequency),
        columns=("Timestamp",))
    # prototype_df['Status'] = 'None'

    # for index in common_gaps_df.index:
    #     gap_start = common_gaps_df.loc[index]['Timestamp']
    #     gap_end = common_gaps_df.loc[index]['Timestamp'] + common_gaps_df.loc[index]['Duration']
    #     gaps_df = prototype_df[prototype_df['Timestamp'] >= gap_start]
    #     gaps_df = gaps_df[gaps_df['Timestamp'] < gap_end]
    #     prototype_df.loc[gaps_df.index, 'Status'] = 'Common gap'

    ret_df = pd.DataFrame()

    sats = df['Satellite'].unique()
    sats.sort()
    for sat in sats:
        sat_df = df[df['Satellite'] == sat]
        sat_df_copy = sat_df.copy()
        sat_df_copy = sat_df_copy.set_index('Timestamp')
        # sat_df_copy['Status'] = 'Data'
        # print(sat_df_copy)
        prototype_df_copy = prototype_df.copy()
        prototype_df_copy = prototype_df_copy.set_index('Timestamp')
        prototype_df_copy = prototype_df_copy.combine_first(sat_df_copy)
        # print(prototype_df_copy)
        prototype_df_copy['Satellite'] = sat
        prototype_df_copy = prototype_df_copy.reset_index()
        # prototype_df_copy.to_csv("prototype.csv")
        # prototype_df_copy['isin'] = prototype_df_copy['Timestamp'].isin(sat_df['Timestamp'])
        # prototype_df_copy.loc[prototype_df_copy[prototype_df_copy['P range'].notnull()].index, 'Status'] = 'Data'
        # prototype_df_copy = prototype_df_copy.drop(columns=['isin',])
        ret_df = pd.concat([ret_df, prototype_df_copy], ignore_index=True)

    return ret_df


def add_elevations(df: pd.DataFrame, xyz: list, nav_path: str, year: int, doy: int, cutoff: float):
    working_df = df.copy()
    working_df['Elevation'] = 'None'

    elevations_for_sat = get_elevations(nav_path, xyz, year, doy, cutoff)

    for sat in working_df['Satellite'].unique():
        if sat in elevations_for_sat.keys():
            sat_df = working_df[working_df['Satellite'] == sat]

            elevation = list(elevations_for_sat[sat])
            # elevation = ['None' if math.isnan(el) else el for el in elevation]

            working_df.loc[sat_df.index, 'Elevation'] = elevation
            # print(working_df[working_df['Elevation'] != 'None'])
            # test = working_df[working_df['Satellite'] == sat]
            # test.to_csv(f'{sat}.csv')
        else:
            print(f'There is no satellite {sat} in elevations list')
    
    return working_df


def get_xyz(file: str):
    with open(file, 'r') as f:
        for i in range(30):
            line = f.readline()
            if 'APPROX POSITION XYZ' in line:
                splited = line.split()
                xyz = splited[0:3]
                xyz = list(map(float, xyz))
                print(splited)
                return xyz


def devide_by_time(df):
    working_df = df[df['Elevation'].notna()]
    diff = working_df["Timestamp"].diff()
    
    borders = diff[diff > pd.Timedelta(30, 'min')]
    print(borders)

    borders_indexes = borders.index

    ret = []

    for index, border in enumerate(borders_indexes):
        if index == 0:
            part = working_df[working_df.index < border]
        else:
            part = working_df[working_df.index >= borders_indexes[index - 1]]
            part = part[part.index < border]

        ret.append(part)
    
    ret.append(working_df[working_df.index > borders_indexes[-1]])

    return ret


def check_phase_tec(df: pd.DataFrame, std_mult: float = 1):
    # считаем производную (дельты)
    # что делаем с пропусками - ???? (пока пропускаем)
    # по дельтам смотрим выбросы - определяем таким образом срыв фазы
    working_df = df[df['Phase tec'].notna()]
    x = working_df['Timestamp'].values
    y = working_df['Phase tec'].values

    y_diff = np.diff(y)
    std_y_diff = np.std(y_diff)

    df_for_plot = pd.DataFrame(zip(x, y_diff), columns=('Timestamp', 'Phase tec'))
    df_for_plot['Color'] = 'Green'
    df_for_plot.loc[df_for_plot[df_for_plot['Phase tec'].abs() > std_y_diff * std_mult].index, 'Color'] = 'Red'

    red_tec = df_for_plot[df_for_plot['Color'] == 'Red']
    ret = list(zip(red_tec['Timestamp'].values, red_tec['Phase tec'].values))

    return ret


def plot_check_phase_tec(df: pd.DataFrame, std_mult: float = 1, sat: str = '', show_plot: bool = False, save_plot: str = None):
    working_df = df[df['Phase tec'].notna()]
    x = working_df['Timestamp'].values
    y = working_df['Phase tec'].values

    y_diff = np.diff(y)
    std_y_diff = np.std(y_diff)

    df_for_plot = pd.DataFrame(zip(x, y_diff), columns=('Timestamp', 'Phase tec'))
    df_for_plot['Color'] = 'Green'
    df_for_plot.loc[df_for_plot[df_for_plot['Phase tec'].abs() > std_y_diff * std_mult].index, 'Color'] = 'Red'

    fig, ax = plt.subplots(nrows=2, ncols=1)
    fig.suptitle(f'{sat} Phase tec')
    ax[0].scatter(x, y)  
    ax[1].scatter(df_for_plot['Timestamp'], df_for_plot['Phase tec'], color=df_for_plot['Color'])
    if show_plot:
        plt.show()
    if save_plot:
        plt.savefig(save_plot)


def check_range_tec(df: pd.DataFrame, poli_degree: int = 1, std_mult: float = 1):
    # заполняем пропуски средними значениями
    # удаляем тренд
    # проверяем СКО
    x = df["Timestamp"].values
    yr = df["P range tec"].values

    mean_yr = np.nanmean(yr)
    yr = [mean_yr if np.isnan(y) else y for y in yr ]

    x_range = range(0, len(x))
    z = np.polyfit(x_range, yr, poli_degree)
    p = np.poly1d(z)

    detrended_yr = yr - p(x_range)
    std_yr = np.std(detrended_yr)

    df_for_plot = pd.DataFrame(zip(x, detrended_yr), columns=('Timestamp', 'P range tec'))

    df_for_plot['Color'] = 'Green'
    df_for_plot.loc[df_for_plot[df_for_plot['P range tec'].abs() > std_yr * std_mult].index, 'Color'] = 'Red'

    red_tec = df_for_plot[df_for_plot['Color'] == 'Red']
    ret = list(zip(red_tec['Timestamp'].values, red_tec['P range tec'].values))

    return ret


def plot_check_range_tec(df: pd.DataFrame, poli_degree: int = 1, std_mult: float = 1, sat: str = '', show_plot: bool = False, save_plot: str = None):
    x = df["Timestamp"].values
    yr = df["P range tec"].values

    mean_yr = np.nanmean(yr)
    yr = [mean_yr if np.isnan(y) else y for y in yr ]

    x_range = range(0, len(x))
    z = np.polyfit(x_range, yr, poli_degree)
    p = np.poly1d(z)

    detrended_yr = yr - p(x_range)
    std_yr = np.std(detrended_yr)

    df_for_plot = pd.DataFrame(zip(x, detrended_yr), columns=('Timestamp', 'P range tec'))

    df_for_plot['Color'] = 'Green'
    df_for_plot.loc[df_for_plot[df_for_plot['P range tec'].abs() > std_yr * std_mult].index, 'Color'] = 'Red'
    
    fig, ax = plt.subplots(nrows=2, ncols=1)
    fig.suptitle(f'{sat} P range tec')
    ax[0].scatter(x, yr)
    ax[0].plot(x, p(x_range), "r--")    
    ax[1].scatter(df_for_plot['Timestamp'], df_for_plot['P range tec'], color=df_for_plot['Color'])
    if show_plot:
        plt.show()
    if save_plot:
        plt.savefig(save_plot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', type=str, nargs='+', help='path to RINEX file')
    parser.add_argument('--interval', type=float, help='interval of RINEX file, in seconds')
    parser.add_argument('--poli-degree', type=int, help='degree of the fitting polynomial')
    parser.add_argument('--std-mult', type=int, help='multiplier of standard deviation')
    parser.add_argument('--plot-show', action='store_true', help='show plot')
    parser.add_argument('--plot-dir', type=str, default=None, help='path to dir to save plot images')
    parser.add_argument('--nav-file', type=str, help='path to NAV file')
    parser.add_argument('--year', type=int, help='Year like 2022')
    parser.add_argument('--doy', type=int, help='Day of year like 103')
    parser.add_argument('--cutoff', type=float, help='Cutoff for elevation')
    args = parser.parse_args()

    interval = timedelta(seconds=args.interval)

    df = pd.DataFrame()
    xyz = get_xyz(args.files[0])

    for file in args.files:
        print(f'Read {file}')
        temp_df = read_to_df(file)
        df = pd.concat([df, temp_df], ignore_index=True)

    print('Find common gaps')
    common_gaps_df = find_common_gaps(df, interval)
    print('Prepare dataframe')
    working_df = prepare_dataframe(df, common_gaps_df, interval)
    print('Add elevations')
    working_df = add_elevations(working_df, xyz, args.nav_file, args.year, args.doy, args.cutoff)

    # working_df = calculate_combinations(working_df)

    # sat_df = working_df[working_df["Satellite"] == "G25"]
    # devided_dfs = devide_by_time(sat_df)

    # for part in devided_dfs:
    #     # check_range_tec(df=part, poli_degree=20)
    #     check_phase_tec(df=part, std_mult=2)

    print('Find problems by satellite')
    phase_tec_problem_by_sat = {}
    range_tec_problem_by_sat = {}

    for sat in working_df['Satellite'].unique():
        print(f'Process {sat}')
        sat_df = working_df[working_df["Satellite"] == sat]
        devided_sat_df = devide_by_time(sat_df)
        range_tec_problem_by_sat[sat] = []
        phase_tec_problem_by_sat[sat] = []
        for index, part in enumerate(devided_sat_df):
            range_tec_problem_by_sat[sat].append(check_range_tec(df=part, poli_degree=args.poli_degree, std_mult=args.std_mult))
            phase_tec_problem_by_sat[sat].append(check_phase_tec(df=part, std_mult=args.std_mult))

            if args.plot_show or args.plot_dir:
                phase_tec_file = None
                range_tec_file = None
                if args.plot_dir:
                    phase_tec_file = os.path.join(args.plot_dir, f"{sat}_phase_tec_{index}.png")
                    range_tec_file = os.path.join(args.plot_dir, f"{sat}_range_tec_{index}.png")
                plot_check_phase_tec(df=part, std_mult=args.std_mult, sat=sat, show_plot=args.plot_show, save_plot=phase_tec_file)
                plot_check_range_tec(df=part, poli_degree=args.poli_degree, std_mult=args.std_mult, sat=sat, show_plot=args.plot_show, save_plot=range_tec_file)

    # pprint(phase_tec_problem_by_sat)
    # pprint(range_tec_problem_by_sat)