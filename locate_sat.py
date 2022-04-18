from pathlib import Path
from coordinates import satellite_xyz
from datetime import datetime, timedelta
import argparse
import numpy as np
import matplotlib.pyplot as plt

from coord import xyz_to_el_az, lle_to_xyz, cart_to_lle




sats = ['G' + str(i).zfill(2) for i in range(1, 33)]
sats.extend(['R' + str(i).zfill(2) for i in range(1, 25)])
sats.extend(['E' + str(i).zfill(2) for i in range(1, 37)])
sats.extend(['C' + str(i).zfill(2) for i in range(1, 41)])
sats.extend(['S' + str(i).zfill(2) for i in range(1, 41)])
sats.extend(['J' + str(i).zfill(2) for i in range(1, 7)])
sats_ind = {sat: i for i, sat in enumerate(sats)}
ind_for_sat = {i: sat for sat, i in sats_ind.items()}
system = ['G', 'R', 'E', 'C']
system_labels = {'G':'GPS', 'R':'GLONASS', 'E':'Galileo', 'C':'COMPASS'}



def get_sat_pos(timestamp, satellite, navs):
    """
    Define satellite position 
    :param timestamp:   datetime.datetime
        Given time
    :param satellite:   str
        Satellite number
    :param navs:    dict
        Navigation files info
    :return:    tuple or None
        Satellite X, Y, Z 
    """
    sat_num = int(satellite[1:])
    gnss_type = satellite[0]
    nav_file = navs.get('rinex3')

    if nav_file is None:
        raise ValueError('No nav file')

    return satellite_xyz(nav_file, gnss_type, sat_num, timestamp)

def locate_sat(navs, start_date, end_date, interval=timedelta(seconds=30)):
    td = end_date - start_date + interval
    assert td.total_seconds() // 3600 <= 24
    tdim = int(td / interval)
    date = start_date
    satdim = len(sats)
    xyz = np.zeros((tdim, satdim, 3))
    times = []
    for i in range(tdim):
        for sat in sats:
            try:
                xyz[i, sats_ind[sat], :] = get_sat_pos(date, sat, navs)
            except IndexError as e:
                pass
                #print(f'{sat} has no records for {date}')
        # print(f'Finished {date}')
        times.append(date)
        date = date + interval
    return xyz, times

def get_elaz(xyz, locs):
    tdim = xyz.shape[0]
    satdim = xyz.shape[1]
    elaz = np.zeros((1, tdim, satdim, 2))
    # for iloc, loc in enumerate(locs):
    loc_xyz = lle_to_xyz(*locs)
    for i in range(tdim):
        for sat in sats:
            isat = sats_ind[sat]
            try:
                elaz[0, i, isat, :] = xyz_to_el_az(loc_xyz, xyz[i, isat])
            except IndexError as e:
                pass
    elaz = np.radians(elaz)
    return elaz


def get_elevations(nav, receiver_xyz, start_date, end_date, interval, cutoff):
    navs = {'rinex3': str(nav)}
    xyz, times = locate_sat(navs, start_date, end_date, interval)
    lle = cart_to_lle(*receiver_xyz)
    elaz = get_elaz(xyz, lle)
    location = 0
    elevation_for_sat = {}
    for isat in range(elaz.shape[2]):
        elevation = elaz[location, :, isat, 0]
        elevation[elevation < np.radians(float(cutoff))] = None
        elevation_for_sat[ind_for_sat.get(isat)] = elevation

    return elevation_for_sat


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nav', type=Path, help='path to NAV file')
    parser.add_argument('--year', type=str, help='Year like 2022')
    parser.add_argument('--doy', type=str, help='Day of year like 103')
    parser.add_argument('--cutoff', type=int, help='Cutoff for elevation')
    args = parser.parse_args()
    get_elevations(args.nav, (np.radians(52), np.radians(104), 0), args.year, args.doy, args.cutoff)
    # navs = {'rinex3': str(args.nav)}
    # date = datetime(int(args.year), 1, 1) + timedelta(int(args.doy) - 1)
    # xyz, times = locate_sat(navs, date)
    # locs = [[np.radians(52), np.radians(104), 0], ]
    # elaz = get_elaz(xyz, locs)
    # location = 0
    # elevation_for_sat = {}
    # for isat in range(elaz.shape[2]):
    #     elevation = elaz[location, :, isat, 0]
    #     elevation[elevation < np.radians(float(args.cutoff))] = None
    #     elevation_for_sat[ind_for_sat.get(isat)] = elevation
    #     plt.scatter(times, isat*0.5 + elevation)
    # plt.show()

    # print(elevation_for_sat)