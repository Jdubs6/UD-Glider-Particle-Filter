import matplotlib.pyplot as plt
import random
import numpy as np
import math
import json
from read_current import get_closest_current


def estimate_positions(initial_positions, heading_angle, t=1, v_kinematic=0.29, n_samples=200, n_g_std = 0.005, n_c_std =0.005):
    '''
    Input: 
        initial_positions: One or multiple initial positions for the glider, size (1,2) or (n_samples,2)
        heading_angle: One of eight available compass heading
        t: time duration is fixed to one hour, since current data will update every one hour
        v_kinematic: glider's default velocity is 0.29 m/s
        n_samples: default number of samples for particle filter is 200
        n_g_std: Glider velocity noise is zero mean Gaussian with default standard deviation of 0.005
        n_c_std: Current velocity noise is zero mean Gaussian with default standard deviation of 0.005

    Getting familiar with other variables: 
        v_g_north: north component of glider velocity
        v_c_east: east component of current velocity
        and so on...

    Returns: estimated positions, (n_samples,2) array
    '''
    
    # Calculate glider's velocity components
    v_g_north = math.sin(heading_angle)*v_kinematic
    v_g_east = math.cos(heading_angle)*v_kinematic

    # Calculate current's velocity components
    v_c_east, v_c_north = get_closest_current(initial_positions[:,0],initial_positions[:,1], t)

    n_g_mean, n_c_mean = 0, 0

    # Add noise
    n_c_east = np.random.normal(n_c_mean, n_c_std, n_samples)
    n_c_north = np.random.normal(n_c_mean, n_c_std, n_samples)
    n_g_east = np.random.normal(n_g_mean, n_g_std, n_samples)
    n_g_north = np.random.normal(n_g_mean, n_g_std, n_samples)

    # Calculate resultant velocity v = v_g + v_c for both components
    v_east = v_g_east + v_c_east + n_c_east + n_g_east
    v_north = v_g_north + v_c_north + n_c_north + n_g_north
    
    dive_duration = t*3600 # in seconds

    # Calculate traveled distance
    d_east = v_east*dive_duration/111139
    d_north = v_north*dive_duration/111139

    # Calculate estimated positions of the glider
    estimated_long = initial_positions[:,0] + d_east #Finds new position, converts into lattitude and longitude again, than adds to new_position list. 
    estimated_lat = initial_positions[:,1] + d_north
    estimated_positions = np.array([estimated_long, estimated_lat]).T

    return estimated_positions


def initialize_particles(initial_position, n_samples=200):
    '''
    Initializes all the particles at the same location, size (8,n_samples,2)
    8 compass directions, 2 coordinates for longitude, latitude
    '''
    initial_positions = np.zeros((8, n_samples, 2))
    initial_positions[:, :, 0] = initial_position[0]
    initial_positions[:, :, 1] = initial_position[1]

    return initial_positions


def run_filter(datetime, duration, startLong=-73.5007, startLat=38.42143, v_kinematic=0.29, n_samples=200, n_g_std=0.005, n_c_std=0.005):
    initial_position = [startLong, startLat]
    estimated_positions = np.zeros((8,200,2))
    
    times = np.arange(1, duration, 1)
    heading_angle = np.arange(0, 6.28, 6.28/8)
    longs = []
    lats = []
    # longs.append([startLong])
    # lats.append([startLat])

    for t in times:
        if t == 1:
            initial_positions = initialize_particles(initial_position, n_samples=n_samples)
            flattened_initial_positions = initial_positions.reshape(np.shape(initial_positions)[0]*np.shape(initial_positions)[1], 2)
            longs = [flattened_initial_positions[:,0].tolist()]
            lats = [flattened_initial_positions[:,1].tolist()]
        else:
            initial_positions = estimated_positions.copy()
 
        for i, theta in enumerate(heading_angle):
            estimated_positions[i,:,:] = estimate_positions(initial_positions[i,:,:], theta, v_kinematic=v_kinematic, n_samples=n_samples, n_g_std=n_g_std, n_c_std=n_c_std)

        flattened_estimated_positions = estimated_positions.reshape(np.shape(estimated_positions)[0]*np.shape(estimated_positions)[1], 2)
        longs.append(flattened_estimated_positions[:,0].tolist())
        lats.append(flattened_estimated_positions[:,1].tolist())

    # pointsJSON = json.dumps({"time": times, "longs": longs, "lats": lats})
    print_dimensions(lats)
    print_dimensions(longs)
    print_dimensions(times)
    return #pointsJSON

def print_dimensions(nested_list, level=0):
    if isinstance(nested_list, list):
        print(f"Dimension {level + 1}: {len(nested_list)}")
        if len(nested_list) > 0 and isinstance(nested_list[0], list):
            print_dimensions(nested_list[0], level + 1)


run_filter(1, 7)

