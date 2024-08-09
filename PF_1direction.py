import matplotlib.pyplot as plt
import random
import numpy as np
import math
from read_current import get_closest_current



def update_weights(particles, measurement, measurement_std=0.001):
    """ 
    Update particle weights based on the measurement.
    """
    distances = np.linalg.norm(particles - measurement, axis=1)
    weights = np.exp(-0.5 * (distances / measurement_std) ** 2) + 1e-6

    return weights / np.sum(weights)  # Normalize weights


def resample(particles, measurement, measurement_std=0.001):
    """ 
    Resample particles based on weights.
    """
    weights = update_weights(particles, measurement, measurement_std)
    indices = np.random.choice(len(particles), len(particles), p=weights)
    return particles[indices]



def estimate_positions(initial_positions, heading_angle, t=1, v_kinematic=0.29, n_samples=200, n_g_std = 0.0001, n_c_std =0.005):
    '''
    Input: 
        initial_positions: One or multiple initial positions for the glider, size (1,2) or (n_samples,2)
        heading_angle: Heading calculated from commanded position user input 
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
    v_g_north = np.sin(heading_angle)*v_kinematic
    v_g_east = np.cos(heading_angle)*v_kinematic

    # Calculate current's velocity components
    v_c_east, v_c_north = 0,0 #get_closest_current(initial_positions[:,0],initial_positions[:,1], t)
    # v_c_east, v_c_north = v_c_east*0.51, v_c_north*0.51
    # print("current:", v_c_east, v_c_north)
    n_g_mean, n_c_mean = 0, 0

    # Add noise
    n_c_east = np.random.normal(n_c_mean, n_c_std, n_samples)
    n_c_north = np.random.normal(n_c_mean, n_c_std, n_samples)
    n_g_east = np.random.normal(n_g_mean, n_g_std, n_samples)
    n_g_north = np.random.normal(n_g_mean, n_g_std, n_samples)

    # Calculate resultant velocity v = v_g + v_c for both components
    v_east = v_g_east + v_c_east + n_c_east + n_g_east
    v_north = v_g_north + v_c_north + n_c_north + n_g_north
    
    dive_duration = 2*3600 # in seconds

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
    Initializes all the particles at the same location, size (1,n_samples,2)
    2 coordinates for longitude, latitude
    '''
    initial_positions = np.zeros((1, n_samples, 2))
    initial_positions[:, :, 0] = initial_position[0]
    initial_positions[:, :, 1] = initial_position[1]

    return initial_positions


def plot_positions(initial_positions, estimated_positions, commanded_position=None, measured_position=None, resampled_positions=None):
    
    plt.scatter(initial_positions[:,:,0], initial_positions[:,:,1], color='red', marker='.', label='Initial position')

    # plt.scatter(estimated_positions[:,:,0], estimated_positions[:,:,1], color='orange', marker='.', label='Estimated positions')
    # if commanded_position:
    #     plt.scatter(commanded_position[0], commanded_position[1], color='yellow', marker='x', label='Commanded position')
    if measured_position:
        plt.scatter(measured_position[0], measured_position[1], color='blue', marker='x', label='Measurement after 2 hours')
    # if resampled_positions:
        plt.scatter(resampled_positions[:,:,0], resampled_positions[:,:,1], color='green', marker='.', label='Resampled positions')


def initialize_particles(initial_position, n_samples=200):
    '''
    Initializes all the particles at the same location, size (1,n_samples,2)
    2 coordinates for longitude, latitude
    '''
    initial_positions = np.zeros((1, n_samples, 2))
    initial_positions[:, :, 0] = initial_position[0]
    initial_positions[:, :, 1] = initial_position[1]

    return initial_positions


def run_filter(duration, startLong=-74.239733, startLat=38.414779, v_kinematic=0.29, n_samples=200, n_g_std=0.005, n_c_std=0.005):
    initial_position = [startLong, startLat]
    estimated_positions = np.zeros((1,200,2))
    commanded_position = [-73.19134, 38.414]
    estimated_positions = np.zeros((1,200,2))
    measured_position = [[-74.23063, 38.415283], [-74.22113, 38.41733], [-74.205475, 38.416848], [-74.186723, 38.411815], [-74.16965, 38.407292], [-74.154815, 38.407554]]
    resampled_positions = np.zeros((1,200,2))

    times = np.arange(0, duration, 2)
            

    plt.figure()
    for t in times:
        if t == 0:
            initial_positions = initialize_particles(initial_position, n_samples=n_samples)
            flattened_initial_positions = initial_positions.reshape(np.shape(initial_positions)[0]*np.shape(initial_positions)[1], 2)
            longs = [flattened_initial_positions[:,0].tolist()]
            lats = [flattened_initial_positions[:,1].tolist()]
        else:
            initial_positions = resampled_positions.copy()

        diff_in_lon = commanded_position[0]-initial_positions[0,:,0] #Difference in lattitude between commanded and inital points, in meters.
        diff_in_lat = commanded_position[1]-initial_positions[0,:,1] #Difference in longitude between commanded and initial points, in meters.
        heading_angle = np.arctan2(diff_in_lat,diff_in_lon)
        print(np.mean(heading_angle))

        estimated_positions[0] = estimate_positions(initial_positions[0], heading_angle, t=t, v_kinematic=0.29)
        if measured_position:
            resampled_positions[0,:,:] = resample(estimated_positions[0], measured_position[t//2])



        plot_positions(initial_positions, estimated_positions, commanded_position=commanded_position, measured_position=measured_position[t//2], resampled_positions=resampled_positions)


    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Estimated Glider Position')
    plt.legend()
    plt.grid()
    plt.show()


run_filter(12)

