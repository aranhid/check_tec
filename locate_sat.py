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

def locate_sat(navs, date, hours=24, tstep=timedelta(0, 30)):
    assert hours <=24
    tdim = int(timedelta(0, 3600) / tstep *hours)
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
        print(f'Finished {date}')
        times.append(date)
        date = date + tstep
    return xyz, times

def get_elaz(xyz, locs):
    tdim = xyz.shape[0]
    satdim = xyz.shape[1]
    elaz = np.zeros((len(locs), tdim, satdim, 2))
    for iloc, loc in enumerate(locs):
        loc_xyz = lle_to_xyz(*loc)
        for i in range(tdim):
            for sat in sats:
                isat = sats_ind[sat]
                try:
                    elaz[iloc, i, isat, :] = xyz_to_el_az(loc_xyz, xyz[i, isat])
                except IndexError as e:
                    pass
    elaz = np.radians(elaz)
    return elaz

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nav', type=Path, help='path to NAV file')
    parser.add_argument('--year', type=str, help='Year like 2022')
    parser.add_argument('--doy', type=str, help='Day of year like 103')
    parser.add_argument('--cutoff', type=int, help='Cutoff for elevation')
    args = parser.parse_args()
    navs = {'rinex3': str(args.nav)}
    date = datetime(int(args.year), 1, 1) + timedelta(int(args.doy) - 1)
    xyz, times = locate_sat(navs, date)
    locs = [[np.radians(52), np.radians(104), 0], ]
    elaz = get_elaz(xyz, locs)
    location = 0
    for isat in range(elaz.shape[2]):
        elevation = elaz[location, :, isat, 0]
        elevation[elevation < np.radians(float(args.cutoff))] = None
        plt.scatter(times, isat*0.5 + elevation)
    plt.show()