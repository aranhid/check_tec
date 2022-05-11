import pandas as pd
from types import MappingProxyType
from gnss_tec import rnx, BAND_PRIORITY
from datetime import timedelta

from locate_sat import get_elevations
from coordinates import retrieve_xyz


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
    prototype_df['Status'] = None

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
        sat_df_copy = sat_df.copy()
        sat_df_copy['Status'] = 'Data'
        sat_df_copy = sat_df_copy.set_index('Timestamp')

        prototype_df_copy = prototype_df.copy()
        prototype_df_copy = prototype_df_copy.set_index('Timestamp')
        prototype_df_copy = prototype_df_copy.combine_first(sat_df_copy)
        prototype_df_copy['Satellite'] = sat
        prototype_df_copy = prototype_df_copy.reset_index()

        ret_df = pd.concat([ret_df, prototype_df_copy], ignore_index=True)

    ret_df.loc[ret_df[ret_df['Status'].isna()].index, 'Status'] = 'None'

    return ret_df


def add_elevations(df: pd.DataFrame, interval: timedelta, xyz: list, nav_path: str, cutoff: float):
    working_df = df.copy()
    dates = pd.to_datetime(working_df['Timestamp'].unique())
    dates = sorted(dates)
    start_date = dates[0].to_pydatetime()
    end_date = dates[-1].to_pydatetime()
    working_df['Elevation'] = None


    elevations_for_sat = get_elevations(nav_path, xyz, start_date, end_date, interval, cutoff)

    for sat in working_df['Satellite'].unique():
        if sat in elevations_for_sat.keys():
            sat_df = working_df[working_df['Satellite'] == sat]
            elevation = list(elevations_for_sat[sat])
            if len(elevation) == len(sat_df):
                working_df.loc[sat_df.index, 'Elevation'] = elevation
            else:
                print('len(elevation) != len(working_df)')
        else:
            print(f'There is no satellite {sat} in elevations list')
    
    return working_df


def get_dataframe(files: list, interval: timedelta, nav_file: str = None, cutoff: float = None):
    df = pd.DataFrame()
    xyz = retrieve_xyz(files[0])

    for file in files:
        print(f'Read {file}')
        temp_df = read_to_df(file)
        df = pd.concat([df, temp_df], ignore_index=True)

    print('Find common gaps')
    common_gaps_df = find_common_gaps(df, interval)
    print('Prepare dataframe')
    working_df = prepare_dataframe(df, common_gaps_df, interval)
    if nav_file:
        print('Add elevations')
        working_df = add_elevations(working_df, interval, xyz, nav_file, cutoff)

    return working_df