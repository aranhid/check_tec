import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from check_tec import plot_check_phase_tec, plot_check_range_tec

# function to update the data
def update_phase(i):
    try:
        satellites_dataframe = pd.read_csv('satellites_dataframe.csv')
        phase_problems = pd.read_csv('problems_phase.csv')
        range_problems = pd.read_csv('problems_range.csv')
    except:
        print('error reading file')

    try:
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
            exit()
    except Exception as e:
        print(e)


fig = plt.figure()
sat = 'G05'
mode = 'phase'

# animate
ani = FuncAnimation(fig, update_phase, interval=100)

# fig.tight_layout()
plt.show()