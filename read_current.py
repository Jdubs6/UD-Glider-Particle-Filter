import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, MultiPoint
from shapely.ops import nearest_points


def get_closest_current(glider_long, glider_lat):

    orig = Point(glider_long, glider_lat)

    ubarEast = np.load('/home/adnana/Dropbox (UFL)/Code/UD-Glider-Particle-Filter/ubarEast.npy')

    vbarNorth = np.load('/home/adnana/Dropbox (UFL)/Code/UD-Glider-Particle-Filter/vbarNorth.npy')

    lats = np.load('/home/adnana/Dropbox (UFL)/Code/UD-Glider-Particle-Filter/lats.npy')

    longs = np.load('/home/adnana/Dropbox (UFL)/Code/UD-Glider-Particle-Filter/longs.npy')


    lats_all = lats.flatten()
    longs_all = longs.flatten()

    ubarEast_all = ubarEast.flatten()
    vbarNorth_all = vbarNorth.flatten()




    points = [Point(lon, lat) for lon, lat in zip(longs_all, lats_all)]

    destinations = MultiPoint(points)

    nearest_geoms = nearest_points(orig, destinations)

    closest_point = np.array([nearest_geoms[1].x, nearest_geoms[1].y])

    indices = np.where(longs_all == closest_point[0])[0]

    vertical_current = vbarNorth_all[indices][0]
    horizontal_current = ubarEast_all[indices][0]

    return horizontal_current,vertical_current