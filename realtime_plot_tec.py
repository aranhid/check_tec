import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pandas.errors import EmptyDataError
from matplotlib.animation import FuncAnimation
from check_tec import plot_check_phase_tec, plot_check_range_tec


def update_phase(i):
    try:
        satellites_dataframe = pd.read_csv('satellites_dataframe.csv')
        if mode == 'phase':
            phase_problems = pd.read_csv('problems_phase.csv')
        else:
            range_problems = pd.read_csv('problems_range.csv')

        sat_df = satellites_dataframe[satellites_dataframe['Satellite'] == sat]
        if not sat_df.empty:
            if mode == 'phase':
                if not phase_problems.empty:
                    sat_phase_problems_df = phase_problems[phase_problems['Satellite'] == sat]
                    plot_check_phase_tec(sat_df, sat_phase_problems_df, fig, sat=sat)
            else:
                if not range_problems.empty:
                    sat_range_problems_df = range_problems[range_problems['Satellite'] == sat]
                    plot_check_range_tec(sat_df, sat_range_problems_df, fig, sat=sat)
        else:
            print(f'There is no data for {sat}, exit.')
            exit()
    except EmptyDataError as e:
        print('error reading file')
    except Exception as e:
        print(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('satellite', type=str, help='The name of the satellite, for example G05')
    parser.add_argument('mode', type=str, default='phase', help='"phase" or "range"')

    args = parser.parse_args()

    global fig, sat, mode
    sat = args.satellite
    mode = args.mode

    fig = plt.figure()
    ani = FuncAnimation(fig, update_phase, interval=100)

    fig.set_tight_layout(True)
    plt.show()


if __name__ == '__main__':
    main()