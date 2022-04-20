import os
import argparse
from pprint import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from gnss_tec import gnss

from reader import get_dataframe


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


def get_windows(data, window, step):
    win = np.arange(window)[None, :]
    shift = np.arange(0, data.shape[0] - win[0][-1], step)[:, None]
    indexer = win + shift
    return indexer


def devide_by_time(df):
    working_df = df[df['Elevation'].notna()]
    diff = working_df["Timestamp"].diff()
    
    borders = diff[diff > pd.Timedelta(60, 'min')]
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
    parser.add_argument('--std-mult-range', type=int, help='multiplier of standard deviation for range TEC')
    parser.add_argument('--std-mult-phase', type=int, help='multiplier of standard deviation for phase TEC')
    parser.add_argument('--plot-show', action='store_true', help='show plot')
    parser.add_argument('--plot-dir', type=str, default=None, help='path to dir to save plot images')
    parser.add_argument('--nav-file', type=str, help='path to NAV file')
    parser.add_argument('--cutoff', type=float, help='Cutoff for elevation')
    args = parser.parse_args()

    interval = timedelta(seconds=args.interval)

    common_gaps_df, working_df = get_dataframe(args.files, interval, args.nav_file, args.cutoff)

    print('Find problems by satellite')
    phase_tec_problem_by_sat = {}
    range_tec_problem_by_sat = {}

    for sat in working_df['Satellite'].unique():
        print(f'Process {sat}')
        sat_df = working_df[working_df["Satellite"] == sat]
        devided_sat_df = devide_by_time(sat_df)
        phase_tec_problem_by_sat[sat] = []
        range_tec_problem_by_sat[sat] = []
        for index, part in enumerate(devided_sat_df):
            phase_tec_problem_by_sat[sat].append(check_phase_tec(df=part, std_mult=args.std_mult_phase))
            range_tec_problem_by_sat[sat].append(check_range_tec(df=part, poli_degree=args.poli_degree, std_mult=args.std_mult_range))

            if args.plot_show or args.plot_dir:
                phase_tec_file = None
                range_tec_file = None
                if args.plot_dir:
                    phase_tec_file = os.path.join(args.plot_dir, f"{sat}_phase_tec_{index}.png")
                    range_tec_file = os.path.join(args.plot_dir, f"{sat}_range_tec_{index}.png")
                plot_check_phase_tec(df=part, std_mult=args.std_mult_phase, sat=sat, show_plot=args.plot_show, save_plot=phase_tec_file)
                plot_check_range_tec(df=part, poli_degree=args.poli_degree, std_mult=args.std_mult_range, sat=sat, show_plot=args.plot_show, save_plot=range_tec_file)

    # pprint(phase_tec_problem_by_sat)
    # pprint(range_tec_problem_by_sat)