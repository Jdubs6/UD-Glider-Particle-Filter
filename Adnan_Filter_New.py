import matplotlib.pyplot as plt
import numpy as np
import math
import json
import sys
sys.path.append(r"C:\Users\27imb\Downloads\read_current.py")
from read_current import get_closest_current

initial_position = [-73.5007, 38.42143]
commanded_position = [-74.00, 39.1669]

pointLongs=[]
pointLats=[]

after_2Hour = [-73.5086, 38.43054]
avg_speed = 0.29  # in m/s
travel_time = 7200  # 2 hours in seconds
number_of_samples = 200

def update_weights(particles, measurement, measurement_std=0.001):
    distances = np.linalg.norm(particles - measurement, axis=1)
    weights = np.exp(-0.5 * (distances / measurement_std) ** 2)
    return weights / np.sum(weights)  # Normalize weights

def resample(particles, weights):
    indices = np.random.choice(len(particles), len(particles), p=weights)
    return particles[indices]

def convert_to_meters(initial_position, commanded_position):
    diff_in_lon = commanded_position[0] - initial_position[0]
    diff_in_lat = commanded_position[1] - initial_position[1]
    angle_of_distance = math.atan2(diff_in_lat, diff_in_lon)
    horizontal_speed = avg_speed * math.cos(angle_of_distance)
    vertical_speed = avg_speed * math.sin(angle_of_distance)
    
    horizontal_current, vertical_current = get_closest_current(initial_position[0], initial_position[1])

    particles = np.zeros((number_of_samples, 2))
    
    for i in range(number_of_samples):
        noise_current_h = np.random.uniform(-0.05, 0.05)
        noise_current_v = np.random.uniform(-0.05, 0.05)
        noise_glider_h = np.random.uniform(-0.05, 0.05)
        noise_glider_v = np.random.uniform(-0.05, 0.05)
        
        net_horizontal_speed = horizontal_speed + horizontal_current + noise_current_h + noise_glider_h
        net_vertical_speed = vertical_speed + vertical_current + noise_current_v + noise_glider_v
        
        distance_traveled_horizontally = net_horizontal_speed * travel_time
        distance_traveled_vertically = net_vertical_speed * travel_time

        particles[i, 0] = initial_position[0] + distance_traveled_horizontally / 111139
        particles[i, 1] = initial_position[1] + distance_traveled_vertically / 111139
    
    weights = update_weights(particles, np.array(after_2Hour))
    resampled_particles = resample(particles, weights)
    
    return resampled_particles

def run_filter(datetime, duration, startLong, startLat, desLong, desLat):
    times=[0]
    pointJSON = json.dumps({"time": times, "longs": pointLongs.tolist(), "lats": pointLats.tolist()})
    return pointJSON

def plot_positions(initial_position, commanded_position):
    resampled_positions = convert_to_meters(initial_position, commanded_position)

    # Extract longitudes and latitudes from resampled positions
    global pointLongs
    pointLongs = resampled_positions[:, 0]
    global pointLats
    pointLats = resampled_positions[:, 1]

    plt.figure()
    plt.scatter(commanded_position[0], commanded_position[1], color='green', marker='x', label='Commanded Position')
    plt.scatter(initial_position[0], initial_position[1], color='red', marker='x', label='Initial Position')
    plt.scatter(after_2Hour[0], after_2Hour[1], color='blue', marker='x', label='Measurement after 2 hours')
    plt.scatter(pointLongs, pointLats, marker='+', color='orange', label='Resampled Positions')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Estimated Glider Position')
    plt.legend()
    plt.grid(True)
    plt.show()
    return(run_filter(None, None, initial_position[0], initial_position[1], commanded_position[0], commanded_position[1]))

plot_positions(initial_position, commanded_position)
