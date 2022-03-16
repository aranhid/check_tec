import os
import argparse
import datetime
import pandas as pd
import plotly.express as px
from types import MappingProxyType
from gnss_tec import rnx, gnss, BAND_PRIORITY


def read_to_df(file: str, interval: datetime.timedelta, band_priority: MappingProxyType = BAND_PRIORITY):
    data = []

    with open(file) as obs_file:
        reader = rnx(obs_file, band_priority=band_priority)
        for observables in reader:
            data.append((
                observables.timestamp,
                observables.satellite,
            ))

    df = pd.DataFrame(data, columns=("Timestamp", "Satellite"))
    df['Timestamp end'] = df['Timestamp'] + interval

    return df


def create_plot(df: pd.DataFrame):
    fig = px.timeline(df, x_start="Timestamp",
                      x_end="Timestamp end", y="Satellite")
    fig.show()


def plot_for_file(file: str, deltatime: datetime.timedelta):
    print(f"Read {file} RINEX to DataFrame")
    df = read_to_df(file, deltatime)

    print("Create plot")
    create_plot(df)


def plot_for_dir(directrory: str, file_extension: str, deltatime: datetime.timedelta):
    for r, d, files in os.walk(directrory):
        for file in files:
            if file.endswith(file_extension):
                plot_for_file(os.path.join(directrory, file), deltatime)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='path to RINEX file')
    parser.add_argument('interval', type=int,
                        help='interval of RINEX file, in seconds')
    args = parser.parse_args()

    plot_for_file(args.file, datetime.timedelta(seconds=args.interval))
