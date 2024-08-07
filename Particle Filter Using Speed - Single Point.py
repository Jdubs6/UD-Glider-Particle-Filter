import matplotlib.pyplot as plt
import random
import numpy as np
import math
from read_current import get_closest_current

initial_position = [-73.5007, 38.42143]
commanded_position = [-74.00, 39.1669]

after_2Hour = [-73.5086, 38.43054]
#actual_position = [-73.5248, 38.45004]
new_position = []
avg_speed = (.29) #in m/s, taken from 3 datapoints
travel_time =(7200) #Glider travels for two hours before surfacing.  
# Note: Some datasets do not have a commanded_position!!!!
number_of_samples = 200
#Time is about 2 hours


def update_weights(particles, measurement, measurement_std=0.001):
    """ Update particle weights based on the measurement. """
    distances = np.linalg.norm(particles - measurement, axis=1)
    weights = np.exp(-0.5 * (distances / measurement_std) ** 2)
    return weights / np.sum(weights)  # Normalize weights

def resample(particles, measurement, measurement_std=0.001):
    """ Resample particles based on weights. """
    weights = update_weights(particles, measurement, measurement_std)
    indices = np.random.choice(len(particles), len(particles), p=weights)
    return particles[indices]



def convert_to_meters(initial_position, commanded_position):
    diff_in_lon=(commanded_position[0]-initial_position[0]) #Difference in lattitude between commanded and inital points, in meters.
    diff_in_lat=(commanded_position[1]-initial_position[1]) #Difference in longitude between commanded and initial points, in meters.
    angle_of_distance=(math.atan2(diff_in_lat,diff_in_lon))
    vertical_speed=(math.sin(angle_of_distance)*avg_speed)#+Current vertical speed, with noise. To be added later. 
    horizontal_speed=(math.cos(angle_of_distance)*avg_speed)#+Current horizontal speed, with noise
    horizontal_current, vertical_current = get_closest_current(initial_position[0],initial_position[1])

    vertical_speed = np.full(number_of_samples, vertical_speed)
    horizontal_speed = np.full(number_of_samples, horizontal_speed)
    vertical_current = np.full(number_of_samples, vertical_current)
    horizontal_current = np.full(number_of_samples, horizontal_current)

    mean, std_dev = 0, 0.05

    # Add uniform noise
    np.random.normal()
    noise_current_h = np.random.uniform(mean, std_dev, number_of_samples)
    noise_current_v = np.random.uniform(mean, std_dev, number_of_samples)
    noise_glider_h = np.random.uniform(mean, std_dev, number_of_samples)
    noise_glider_v = np.random.uniform(mean, std_dev, number_of_samples)

    net_horizontal_speed = horizontal_speed + horizontal_current + noise_current_h + noise_glider_h
    net_vertical_speed = vertical_speed + vertical_current + noise_current_v + noise_glider_v
    
    distance_traveled_vertically=(vertical_speed*travel_time)
    distance_traveled_horizontally=(horizontal_speed*travel_time)

    # Calculate possible positions of the glider
    new_position.append(initial_position[0]+(((net_horizontal_speed)*travel_time))/111139) #Finds new position, converts into lattitude and longitude again, than adds to new_position list. 
    new_position.append(initial_position[1]+(((net_vertical_speed)*travel_time))/111139)
    print(len(new_position))

    # Update and resample the possible positions using a new measurement
    resampled_new_positions = resample(np.array(new_position).T, np.array(after_2Hour))

    return resampled_new_positions
    
def calculate_weights(combined_positions):
    position_weights = []
    for position in combined_positions:
        distance = np.sqrt((abs(position[0])-abs(commanded_position[0]))**2+(abs(position[1])-abs(commanded_position[1]))**2) #Distance between actual particle and estimated waypoint.
        position_weight = np.exp(-distance / 0.0005) #0.0005 can be adjusted based on weight sensitivity. 
        position_weights.append(position_weight)
    position_weights = np.array(position_weights)
    position_weights /= np.sum(position_weights) #Normalizing the weights
    resampling = np.random.choice(range(number_of_samples),size=number_of_samples,replace=True,p=position_weights)
    resampled_positions = [combined_positions[idx] for idx in resampling]
    return(resampled_positions)
def estimate_positions(commanded_position):
    combined_positions = []
    new_long = []
    new_lat = []

    return((combined_positions)) 
def plot_positions(initial_position, commanded_position):
    resampled_new_positions = convert_to_meters(initial_position, commanded_position)
    print(resampled_new_positions)
    combined_positions = estimate_positions(new_position)
    plt.figure()
    """
    for position in combined_positions:
        plt.scatter(position[0], position[1], color='blue', alpha=0.5)
    """
    plt.scatter(commanded_position[0], commanded_position[1], color='green', marker='x', label='Commanded Position')
    plt.scatter(initial_position[0], initial_position[1], color='red', marker='x', label='Initial Position')
    plt.scatter(new_position[0], new_position[1], color='orange', marker='x', label='New Position')
    plt.scatter(resampled_new_positions[:,0], resampled_new_positions[:,1], marker='+', label='Resampled new Position')
    plt.scatter(after_2Hour[0], after_2Hour[1], color='blue', marker='x', label='Measurement after 2 hour')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Estimated Glider Position')
    plt.legend()
    plt.grid(True)
    plt.show()
    """"
    plt.figure()
    for position in combined_positions:
        plt.scatter(position[0], position[1], color='blue', alpha=0.5)
    plt.scatter(new_position[0], new_position[1], color='orange', marker='x', label='New Position')
    plt.show()
    """
plot_positions(initial_position,commanded_position)