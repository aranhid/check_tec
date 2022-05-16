import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
from gnss_tec import gnss
from datetime import timedelta
from matplotlib.figure import Figure

from reader import get_dataframe

import logging
logging.getLogger('PIL').setLevel(logging.WARNING)


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
    if working_df.empty:
        return []
    
    diff = working_df["Timestamp"].diff()
    
    borders = diff[diff > pd.Timedelta(60, 'min')]
    if borders.empty:
        return [working_df]
    
    borders_indexes = borders.index

    ret = []

    for index, border in enumerate(borders_indexes):
        if index == 0:
            part = working_df[working_df.index < border]
        else:
            part = working_df[working_df.index >= borders_indexes[index - 1]]
            part = part[part.index < border]

        ret.append(part)
    
    if len(borders_indexes):
        ret.append(working_df[working_df.index > borders_indexes[-1]])

    return ret


def check_phase_tec(df: pd.DataFrame, std_mult: float = 1, poli_degree: int = 7, rate: float = 0.035, min_win_size: int = 20, max_win_size: int = 100):
    working_df = df[df['Phase tec'].notna()]
    if len(working_df) == 0:
        return pd.DataFrame()
    x = working_df['Timestamp'].values
    y = working_df['Phase tec'].values
    x = x.astype('datetime64')
    y = y.astype('float64')

    x_diff = np.diff(x)
    x_diff = x_diff / np.timedelta64(1, 's')
    y_diff = np.diff(y)

    tec_diff_by_sec = y_diff / x_diff

    if len(tec_diff_by_sec) < poli_degree:
        return pd.DataFrame()

    x_range = list(range(0, len(x[1:])))
    z = np.polyfit(x_range, tec_diff_by_sec, poli_degree)
    p = np.poly1d(z)

    detrend_tec_diff = tec_diff_by_sec - p(x_range)

    df_for_win = pd.DataFrame(zip(x[1:], detrend_tec_diff), columns=('Timestamp', 'Phase tec'))
    df_for_win['Color'] = 'green'

    for window in df_for_win.rolling(window=max_win_size):
        if (min_win_size <= len(window) <= max_win_size):
            check_win = df_for_win.loc[window.index]
            check_win = check_win[check_win['Color'] != 'red']
            # check_win = window
            win_std = check_win['Phase tec'].std()
            problems = check_win[check_win['Phase tec'].abs() > win_std * std_mult]
            problems = problems[problems['Phase tec'].abs() > rate]
            df_for_win.loc[problems.index, 'Color'] = 'red'

    return df_for_win


def plot_check_phase_tec(part: pd.DataFrame, checked_part: pd.DataFrame, figure: Figure = None, poli_degree: int = 7, sat: str = '', show_plot: bool = False, save_plot: str = None):
    part_phase_tec = part[part['Phase tec'].notna()]
    if len(part_phase_tec) == 0:
        return
    x = part_phase_tec['Timestamp'].values
    y = part_phase_tec['Phase tec'].values
    x = x.astype('datetime64')
    y = y.astype('float64')

    x_diff = np.diff(x)
    x_diff = x_diff / np.timedelta64(1, 's')
    y_diff = np.diff(y)

    tec_diff_by_sec = y_diff / x_diff

    if len(tec_diff_by_sec) < poli_degree:
        return

    x_range = list(range(0, len(x[1:])))
    z = np.polyfit(x_range, tec_diff_by_sec, poli_degree)
    p = np.poly1d(z)
    
    print('Plot phase tec')

    if figure != None:
        fig = figure
        fig.clf()
        ax = fig.subplots(nrows=3, ncols=1)
    else:
        fig, ax = plt.subplots(nrows=3, ncols=1)
    
    fig.suptitle(f'{sat} Phase TEC')
    
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('TECu')
    ax[0].scatter(x, y, color='tab:blue')
    ax[0].tick_params(axis='y', labelcolor='tab:blue')
    ax[0].xaxis.set_tick_params(labelsize=5)

    if "Elevation" in part_phase_tec.columns:
        el = part_phase_tec['Elevation'].values
        ax01 = ax[0].twinx()
        ax01.set_ylabel('Elevation')
        ax01.plot(x, el, linestyle="--", color='tab:orange')
        ax01.tick_params(axis='y', labelcolor='tab:orange')
        ax01.xaxis.set_tick_params(labelsize=5)

    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('TECu/sec')
    ax[1].scatter(x[1:], tec_diff_by_sec)
    ax[1].plot(x[1:], p(x_range), "r--")
    ax[1].xaxis.set_tick_params(labelsize=5)

    ax[2].set_xlabel('Time')
    ax[2].set_ylabel('TECu/sec')
    ax[2].scatter(checked_part['Timestamp'], checked_part['Phase tec'], color=checked_part['Color'])
    ax[2].xaxis.set_tick_params(labelsize=5)

    if show_plot:
        if figure != None:
            fig.canvas.draw()
            fig.canvas.flush_events()
        else:
            plt.show()
    if save_plot:
        plt.savefig(save_plot, dpi=1080/fig.get_size_inches()[1])


def check_range_tec(df: pd.DataFrame, poli_degree: int = 10, std_mult: float = 1, min_win_size: int = 20, max_win_size: int = 100):
    working_df = df[df['P range tec'].notna()]
    if len(working_df) == 0:
        return pd.DataFrame()
    x = working_df["Timestamp"].values
    yr = working_df["P range tec"].values
    x = x.astype('datetime64')
    yr = yr.astype('float64')

    # mean_yr = np.nanmean(yr)
    # yr = [mean_yr if np.isnan(y) else y for y in yr ]

    if len(yr) < poli_degree:
        return pd.DataFrame()

    x_range = range(0, len(x))
    z = np.polyfit(x_range, yr, poli_degree)
    p = np.poly1d(z)

    detrended_yr = yr - p(x_range)

    df_for_win = pd.DataFrame(zip(x, detrended_yr), columns=('Timestamp', 'P range tec'))
    df_for_win['Color'] = 'green'

    for window in df_for_win.rolling(window=max_win_size):
        if (min_win_size <= len(window) <= max_win_size):
            check_win = df_for_win.loc[window.index]
            check_win = check_win[check_win['Color'] != 'red']
            win_std = check_win['P range tec'].std()
            problems = check_win[check_win['P range tec'].abs() > win_std * std_mult]
            df_for_win.loc[problems.index, 'Color'] = 'red'
    
    return df_for_win

def plot_check_range_tec(part: pd.DataFrame, checked_part: pd.DataFrame, figure: Figure = None, poli_degree: int = 10, sat: str = '', show_plot: bool = False, save_plot: str = None):
    part_range_tec = part[part['P range tec'].notna()]
    if len(part_range_tec) == 0:
        return None
    x = part_range_tec["Timestamp"].values
    yr = part_range_tec["P range tec"].values
    x = x.astype('datetime64')
    yr = yr.astype('float64')

    # mean_yr = np.nanmean(yr)
    # yr = [mean_yr if np.isnan(y) else y for y in yr ]

    if len(yr) < poli_degree:
        return None

    x_range = range(0, len(x))
    z = np.polyfit(x_range, yr, poli_degree)
    p = np.poly1d(z)

    print("Plot range tec")

    if figure != None:
        fig = figure
        fig.clf()
        ax = fig.subplots(nrows=2, ncols=1)
    else:
        fig, ax = plt.subplots(nrows=2, ncols=1)

    fig.suptitle(f'{sat} P range tec')

    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('TECu')
    ax[0].scatter(x, yr)
    ax[0].plot(x, p(x_range), "r--")
    ax[0].xaxis.set_tick_params(labelsize=5)

    if "Elevation" in part_range_tec.columns:
        el = part_range_tec["Elevation"].values
        ax01 = ax[0].twinx()
        ax01.set_ylabel('Elevation')
        ax01.plot(x, el, linestyle="--", color='tab:orange')
        ax01.tick_params(axis='y', labelcolor='tab:orange')
        ax01.xaxis.set_tick_params(labelsize=5)
    
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('TECu')
    ax[1].scatter(checked_part['Timestamp'], checked_part['P range tec'], color=checked_part['Color'])
    ax[1].xaxis.set_tick_params(labelsize=5)

    if show_plot:
        if figure != None:
            fig.canvas.draw()
            fig.canvas.flush_events()
        else:
            plt.show()
    if save_plot:
        plt.savefig(save_plot, dpi=1080/fig.get_size_inches()[1])


def realtime_check_sat(df: pd.DataFrame, fig_phase, fig_range, sat: str = None):
    phase_tec_problems = []
    range_tec_problems = []

    checked_phase = check_phase_tec(df)
    if not checked_phase.empty:
        red_phase_tec = checked_phase[checked_phase['Color'] == 'Red']
        phase_tec_problems = list(zip(red_phase_tec['Timestamp'].values, red_phase_tec['Phase tec'].values))

    checked_range = check_range_tec(df)
    if not checked_range.empty:
        red_range_tec = checked_range[checked_range['Color'] == 'Red']
        range_tec_problems = list(zip(red_range_tec['Timestamp'].values, red_range_tec['P range tec'].values))

    plot_check_phase_tec(df, checked_phase, fig_phase, sat=sat)
    plot_check_range_tec(df, checked_range, fig_range, sat=sat)

    print(sat)
    pprint(phase_tec_problems)
    pprint(range_tec_problems)
    print(" ")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', type=str, nargs='+', help='path to RINEX file')
    parser.add_argument('--interval', type=float, help='interval of RINEX file, in seconds')
    parser.add_argument('--poli-degree-phase', type=float, default=7, help='degree of the fitting polynomial')
    parser.add_argument('--poli-degree-range', type=float, default=10, help='degree of the fitting polynomial')
    parser.add_argument('--std-mult-phase', type=float, default=3, help='multiplier of standard deviation for phase TEC')
    parser.add_argument('--std-mult-range', type=float, default=3, help='multiplier of standard deviation for range TEC')
    parser.add_argument('--phase_rate', type=float, default=0.035, help='multiplier of standard deviation for range TEC')
    parser.add_argument('--min-win-size', type=int, default=20, help='minimum size of window, in measurement')
    parser.add_argument('--max-win-size', type=int, default=100, help='maximum size of window, in measurement')
    parser.add_argument('--plot-show', action='store_true', help='show plot')
    parser.add_argument('--plot-dir', type=str, default=None, help='path to dir to save plot images')
    parser.add_argument('--nav-file', type=str, help='path to NAV file')
    parser.add_argument('--cutoff', type=float, help='Cutoff for elevation')
    args = parser.parse_args()

    interval = timedelta(seconds=args.interval)

    working_df = get_dataframe(args.files, interval, args.nav_file, args.cutoff)

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
            if len(part):
                checked_part_phase = check_phase_tec(df=part, std_mult=args.std_mult_phase, poli_degree=args.poli_degree_phase, rate=args.phase_rate, min_win_size=args.min_win_size, max_win_size=args.max_win_size)
                if not checked_part_phase.empty:
                    red_phase_tec = checked_part_phase[checked_part_phase['Color'] == 'Red']
                    phase_tec_problems = list(zip(red_phase_tec['Timestamp'].values, red_phase_tec['Phase tec'].values))
                    if len(phase_tec_problems):
                        phase_tec_problem_by_sat[sat].append(phase_tec_problems)

                checked_part_range = check_range_tec(df=part, poli_degree=args.poli_degree_range, std_mult=args.std_mult_phase, min_win_size=args.min_win_size, max_win_size=args.max_win_size)
                if not checked_part_range.empty:
                    red_range_tec = checked_part_range[checked_part_range['Color'] == 'Red']
                    range_tec_problems = list(zip(red_range_tec['Timestamp'].values, red_range_tec['P range tec'].values))
                    if len(range_tec_problems):
                        range_tec_problem_by_sat[sat].append(range_tec_problems)

                if args.plot_show or args.plot_dir:
                    phase_tec_file = None
                    range_tec_file = None
                    if args.plot_dir:
                        if not os.path.exists(args.plot_dir):
                            os.makedirs(args.plot_dir)
                        phase_tec_file = os.path.join(args.plot_dir, f"{sat}_phase_tec_{index}.png")
                        range_tec_file = os.path.join(args.plot_dir, f"{sat}_range_tec_{index}.png")
                    if not checked_part_phase.empty:
                        plot_check_phase_tec(part, checked_part_phase, poli_degree=args.poli_degree_phase, sat=sat, show_plot=args.plot_show, save_plot=phase_tec_file)

                    if not checked_part_range.empty:
                        plot_check_range_tec(part, checked_part_range, poli_degree=args.poli_degree_range, sat=sat, show_plot=args.plot_show, save_plot=range_tec_file)

    # pprint(phase_tec_problem_by_sat)
    # pprint(range_tec_problem_by_sat)