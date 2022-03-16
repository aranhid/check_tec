import os
import plotly.express as px
import datetime
from types import MappingProxyType
from gnss_tec import rnx, gnss, BAND_PRIORITY
import pandas as pd


def read_to_df(file: str, timedelta: datetime.timedelta, band_priority: MappingProxyType = BAND_PRIORITY):
    data = []

    with open(file) as obs_file:
        reader = rnx(obs_file, band_priority=band_priority)
        for observables in reader:
            # if observables.timestamp < datetime.datetime(year=2022, month=3, day=9, hour=7, minute=37):
            #     continue
            # if observables.timestamp >  datetime.datetime(year=2022, month=3, day=9, hour=7, minute=38):
            #     break
            data.append((
                observables.timestamp,
                observables.timestamp + timedelta,
                observables.satellite,
                # observables.phase_code,
                # observables.phase,
                # observables.phase_tec,
                # observables.p_range_code,
                # observables.p_range,
                # observables.p_range_tec,
            ))

    df = pd.DataFrame(data, columns=("Timestamp", "Timestamp end", "Satellite"))
    # df = pd.DataFrame(data, columns=("Timestamp", "Timestamp end", "Satellite", "Phase code", "Phase", "Phase tec", "P range code", "P range", "P range tec"))

    return df

def create_plot(df: pd.DataFrame):
    fig = px.timeline(df, x_start="Timestamp", x_end="Timestamp end", y="Satellite")
    fig.show()

def plot_for_file(file: str, deltatime: datetime.timedelta):
    print(f"Read {file} RINEX to DataFrame")
    df = read_to_df(file, deltatime)

    print("Create plot")
    create_plot(df)

def plot_for_dir(directrory: str, file_extension: str, deltatime: datetime.timedelta):
    for r,d,files in os.walk(directrory):
        for file in files:
            if file.endswith(file_extension):
                plot_for_file(os.path.join(directrory, file), deltatime)
                

def find_gap(df: pd.DataFrame):
    diff = df["Timestamp"].diff()
    print(diff)


# Критерии: 
# 1)+ Длина пропуска (посчитать для всех пропусков)
# 2) Периодичность пропусков
# 3) Плотность пропусков (количество пропусков за время) (почитать время между пропусками, отфильтровать по максимальному временному окну)

def find_gap_in_sat(df: pd.DataFrame, sat: str, deltatime: datetime.timedelta):
    sat_df = df[df['Satellite'] == sat]
    time_diff = sat_df['Timestamp'].diff()
    dt = pd.Timedelta(deltatime)
    problem_time_diff = time_diff[time_diff > dt]
    print(time_diff)
    print(problem_time_diff)
    gap_times = []
    for index in problem_time_diff.index:
        gap_time = sat_df.loc[index]["Timestamp"]
        gap_times.append((gap_time - deltatime, problem_time_diff.loc[index]))
        # print(gap_time - deltatime)
        # print(sat_df.loc[index])
        # print("----")
    print(gap_times)


obs_file1 = 'observables\IST5063S.22O'
obs_file2 = 'observables\IST5063T.22O'
obs_file3 = 'observables\IST5063U.22O'
obs_file4 = 'observables\IST5063V.22O'
obs_file5 = 'observables\IST5063W.22O'
obs_file6 = 'observables\IST5063X.22O'
obs_file7 = 'observables\IST5064A.22O'
obs_file8 = 'observables\IST5064B.22O'

obs_file5_20msec = 'observables\SEPT0630W.22O'
obs_file6_20msec = 'observables\SEPT0630X.22O'

MY_BAND_PRIORITY = MappingProxyType({
    gnss.GPS: ((1, 2), ),
    gnss.GLO: ((2, 2), ),
    gnss.GAL: (),
    gnss.SBAS: (),
    gnss.QZSS: (),
    gnss.BDS: (),
    gnss.IRNSS: (),
})

CURRENT_DIR = 'C:\\Users\\vladm\\Documents\\Работа\\chart +\\march9'
PATTERN_END = '.22O'

if __name__ == '__main__':
    # dt = datetime.timedelta(milliseconds=20)
    # dt = datetime.timedelta(milliseconds=40)
    dt = datetime.timedelta(seconds=30)
    file2 = 'C:\\Users\\vladm\\Documents\\Работа\\chart +\\observables\\IST5062F.22O'
    file3 = "C:\\Users\\vladm\\Documents\\Работа\\chart +\\observables\\novatel\\ISNO_22MAR10_045948.22O"
    file4 = "C:\\Users\\vladm\\Documents\\Работа\\chart +\\observables\\IST5069A.22O"
    file5 = "C:\\Users\\vladm\\Documents\\Работа\\chart +\\observables\\IST5069B.22O"

    file100HZ = "C:\\Users\\vladm\\Documents\\Работа\\chart +\\100H\\IST5069H.22O"

    novatel1 = "C:\\Users\\vladm\\Documents\\Работа\\chart +\\observables\\novatel\\ISNO_22MAR11_045941.22O"
    novatel2 = "C:\\Users\\vladm\\Documents\\Работа\\chart +\\observables\\novatel\\ISNO_22MAR11_035941.22O"
    novatel3 = "C:\\Users\\vladm\\Documents\\Работа\\chart +\\observables\\novatel\\ISNO_22MAR11_025941.22O"

    march11_13 = "R:\\Septentrio\\22070\\IST5070F.22O"
    march11_14 = "R:\\Septentrio\\22070\\IST5070G.22O"
    march11_15 = "R:\\Septentrio\\22070\\IST5070H.22O"
    novatel_march14_15 = "R:\\NovAtel\\Converted\\ISNO_22MAR14_065942.22O"
    septentrio_march14_5 = "R:\\Septentrio\\22072\\IST5072V.22O"
    plot_for_file(septentrio_march14_5, dt)
    # df = read_to_df(obs_file5, dt)
    # df = read_to_df("C:\\Users\\vladm\\Documents\\Работа\\chart +\\march9\\SEPT0680.22O", dt)
    # find_gap_in_sat(df, "S44", dt)