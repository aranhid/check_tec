#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 12:29:24 2018

@author: Artem Vesnin
"""

from numpy import cos, sin, sqrt, arctan, arctan2, arcsin, pi, rad2deg, inf


RE = 6378000.0  # in meters


def xyz_to_el_az(xyz_site, xyz_sat):
    """compute_el_az(obs, sat) -> el, az
    computes elevation and azimuth to satellite.

    @author: Ilya Zhuvetiev

    Parameters
    ----------
    obs : tuple
        obs = (x, y, z); cartesian coordinates of the observer
    sat : tuple
        sat = (x, y, z); cartesian coordinates of the satellite
    Returns
    -------
    el : float
        elevation, deg
    az : float
        azimuth, deg
    """

    (x_0, y_0, z_0) = xyz_site
    (x_s, y_s, z_s) = xyz_sat

    (b_0, l_0) = cart_to_lle(*xyz_site)[0:2]
    (b_s, l_s) = cart_to_lle(*xyz_sat)[0:2]

    r_k = sqrt(x_s ** 2 + y_s ** 2 + z_s ** 2)

    sigma = arctan2(
        sqrt(1 - (
            sin(b_0) * sin(b_s) + cos(b_0) * cos(b_s) * cos(l_s - l_0)) ** 2),
         (sin(b_0) * sin(b_s) + cos(b_0) * cos(b_s) * cos(l_s - l_0))
    )

    x_t = -(x_s - x_0) * sin(l_0) + (y_s - y_0) * cos(l_0)
    y_t = (-(x_s - x_0) * cos(l_0) * sin(b_0) -
           (y_s - y_0) * sin(l_0) * sin(b_0) + (z_s - z_0) * cos(b_0))

    el = arctan2((cos(sigma) - RE / r_k), sin(sigma))
    az = arctan2(x_t, y_t)

    el = rad2deg(el)
    az = rad2deg(az)

    # 0 - 360
    if az < 0:
        az += 360

    return el, az


def _xyz_to_el_az(xyz_site, xyz_sat):
    (x_0, y_0, z_0) = xyz_site
    (x_s, y_s, z_s) = xyz_sat
    x_s = xyz_sat[:, 0]
    y_s = xyz_sat[:, 1]
    z_s = xyz_sat[:, 2]

    (b_0, l_0) = cart_to_lle(*xyz_site)[0:2]
    (b_s, l_s) = cart_to_lle(*xyz_sat)[0:2]

    r_k = sqrt(x_s ** 2 + y_s ** 2 + z_s ** 2)

    sigma = arctan2(
        sqrt(1 - (
            sin(b_0) * sin(b_s) + cos(b_0) * cos(b_s) * cos(l_s - l_0)) ** 2),
         (sin(b_0) * sin(b_s) + cos(b_0) * cos(b_s) * cos(l_s - l_0))
    )

    x_t = -(x_s - x_0) * sin(l_0) + (y_s - y_0) * cos(l_0)
    y_t = (-(x_s - x_0) * cos(l_0) * sin(b_0) -
           (y_s - y_0) * sin(l_0) * sin(b_0) + (z_s - z_0) * cos(b_0))

    el = arctan2((cos(sigma) - RE / r_k), sin(sigma))
    az = arctan2(x_t, y_t)

    el = rad2deg(el)
    az = rad2deg(az)

    # 0 - 360
    if az < 0:
        az += 360

    return el, az

def lle_to_xyz(lat, lon, altitude=0):
    x = (altitude + RE) * cos(lat) * cos(lon)
    y = (altitude + RE) * cos(lat) * sin(lon)
    z = (altitude + RE) * sin(lat)
    return x, y, z
    


def cart_to_gsc(x, y, z):
    """cart_to_gsc(x, y, z)
    output radius from center, phi (longitude), theta(latitude)
    Transform of cartsian coordinates to GSC
    The angles are given in radians
    """
    r = sqrt(x ** 2 + y ** 2 + z ** 2)
    t = y / x if x != 0 else inf
    l = arctan(t)
    if x < 0 and y >= 0:
        l = l + pi
    if x < 0 and y < 0:
        l = l - pi
    return r, l, arcsin(z / r)


def cart_to_lle(x, y, z):
    """cart_to_lle(x, z, y)
    output latitude, longitude, elevation,
    Transform of cartsian coordinates to Latitude, Longitude and Elevation.
    The angles are given in radians
    """
    r, lon, lat = cart_to_gsc(x, y, z)
    return lat, lon, r - RE


def __convert_to(xx, yy, zz, transform):
    xx_, yy_, zz_ = [], [], []
    if len(xx) != len(yy) != len(zz):
        raise ValueError("Array for x, y and z should be same length " +
                         "len(xx) = %d, len(yy) = %d len(zz) = %d. " %
                         (len(xx), len(yy), len(zz)))
    for i in range(0, len(xx)):
        x_, y_, z_ = transform(xx[i], yy[i], zz[i])
        xx_.append(x_)
        yy_.append(y_)
        zz_.append(z_)
    return xx_, yy_, zz_


def convert_to_gsc(xx, yy, zz):
    """convert_to_gsc(xx, yy, zz)
    output lists of GSC coordinates
    Converts input arrays of x,y,z to GSC
    """
    return __convert_to(xx, yy, zz, cart_to_gsc)


def convert_to_lle(xx, yy, zz):
    """convert_to_lle(xx, yy, zz)
    output lists of GSC coordinates
    Converts input arrays of x,y,z to Latitude, Longitude and Elevation
    """
    return __convert_to(xx, yy, zz, cart_to_lle)
