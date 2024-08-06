import matplotlib.pyplot as plt
import random
import numpy as np
import math

initial_position = [-73.5007, 38.42143]
commanded_position = [-74.00, 39.1669]
#actual_position = [-73.5248, 38.45004]
new_position = []
avg_speed = (.29) #in m/s, taken from 3 datapoints
travel_time =(7200) #Glider travels for two hours before surfacing.  
# Note: Some datasets do not have a commanded_position!!!!
number_of_samples = 200
#Time is about 2 hours
def convert_to_meters(initial_position, commanded_position):
    diff_in_lon=(commanded_position[0]-initial_position[0]) #Difference in lattitude between commanded and inital points, in meters.
    diff_in_lat=(commanded_position[1]-initial_position[1]) #Difference in longitude between commanded and initial points, in meters.
    angle_of_distance=(math.atan2(diff_in_lat,diff_in_lon))
    vertical_speed=(math.sin(angle_of_distance)*avg_speed)#+Current vertical speed, with noise. To be added later. 
    horizontal_speed=(math.cos(angle_of_distance)*avg_speed)#+Current horizontal speed, with noise
    
    distance_traveled_vertically=(vertical_speed*travel_time)
    distance_traveled_horizontally=(horizontal_speed*travel_time)
    new_position.append(initial_position[0]+(((horizontal_speed)*travel_time))/111139) #Finds new position, converts into lattitude and longitude again, than adds to new_position list. 
    new_position.append(initial_position[1]+(((vertical_speed)*travel_time))/111139)
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
    get_new_position=convert_to_meters(initial_position, commanded_position)
    combined_positions = estimate_positions(new_position)
    plt.figure()
    """
    for position in combined_positions:
        plt.scatter(position[0], position[1], color='blue', alpha=0.5)
    """
    plt.scatter(commanded_position[0], commanded_position[1], color='green', marker='x', label='Commanded Position')
    plt.scatter(initial_position[0], initial_position[1], color='red', marker='x', label='Initial Position')
    plt.scatter(new_position[0], new_position[1], color='orange', marker='x', label='New Position')
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