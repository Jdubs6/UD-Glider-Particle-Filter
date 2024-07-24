"""
Author: Adnan Abdullah
Date: 07-23-2024
Description:
            This script reads a csv file of glider's location (latitude,longitude).
            'latitude' and 'langitude' columns hold the commanded positions.
            'lat_uv' and 'lon_uv' columns hold the actual positions.
            It applies a particle filter to estimate possible new locations.
            Then it plots the locations. 
Usage: python ParticleFilterPart3.py
    or python ParticleFilterPart3.py <path_to_csv_file>
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import sys


def calculate_weights(combined_positions, actual_position, number_of_samples = 200):
    '''
    Applies exponential weight on the estimations
    resamples and keeps number_of_samples
    Returns: resampled positions
    '''
    position_weights = []
    for position in combined_positions:
        distance = np.sqrt((abs(position[0])-abs(actual_position[0]))**2+(abs(position[1])-abs(actual_position[1]))**2)
        position_weight = np.exp(-distance / 0.0005)
        position_weights.append(position_weight)
    position_weights = np.array(position_weights)
    position_weights /= np.sum(position_weights)
    resampling = np.random.choice(range(number_of_samples), size=number_of_samples, replace=True, p=position_weights)
    resampled_positions = [combined_positions[idx] for idx in resampling]
    return resampled_positions


def estimate_positions(commanded_position, actual_position, number_of_samples = 200):
    '''
    Takes a commanded position and corresponding actual position
    Returns: weighted estimated positions
    '''
    combined_positions = []
    new_long = []
    new_lat = []
    estimated_positions_long = [commanded_position[0]] * number_of_samples
    estimated_positions_lat = [commanded_position[1]] * number_of_samples
    for position in estimated_positions_long:
        new_long.append(position + (1.5*random.uniform(-abs(abs(commanded_position[0])-abs(actual_position[0])),abs(abs(commanded_position[0])-abs(actual_position[0])))))
    for position in estimated_positions_lat:
        new_lat.append(position + (1.5*random.uniform(-abs(abs(commanded_position[1])-abs(actual_position[1])),abs(abs(commanded_position[1])-abs(actual_position[1])))))
    for a, b in zip(new_long, new_lat):
        combined_positions.append([a, b])
    return calculate_weights(combined_positions, actual_position)


def plot_positions(commanded_position, actual_position, prev_position):
    '''
    Plots the positions w.r.t. latitudes and longitudes
    '''
    combined_positions = estimate_positions(commanded_position, actual_position)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    # Subplot-1
    for position in combined_positions:
        ax1.scatter(position[0], position[1], color='blue', alpha=0.5)
    ax1.scatter(commanded_position[0], commanded_position[1], color='green', marker='x', label='Commanded Position')
    ax1.scatter(actual_position[0], actual_position[1], color='orange', marker='x', label='Actual Position')
    ax1.scatter(prev_position[0], prev_position[1], color='red', marker='x', label='Previous Position')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title('Estimated Glider Position')
    ax1.legend()
    ax1.grid()

    # Subplot-2
    for position in combined_positions:
        ax2.scatter(position[0], position[1], color='blue', alpha=0.5)
    ax2.scatter(commanded_position[0], commanded_position[1], color='green', marker='x', label='Commanded Position')
    ax2.scatter(actual_position[0], actual_position[1], color='orange', marker='x', label='Actual Position')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    # Set x and y limit for zoomed-in plot
    min_latitude = min(position[0] for position in combined_positions)
    min_longitude = min(position[1] for position in combined_positions)
    max_latitude = max(position[0] for position in combined_positions)
    max_longitude = max(position[1] for position in combined_positions)
    ax2.set_xlim(min_latitude-0.00001, max_latitude+0.00001)
    ax2.set_ylim(min_longitude-0.00001, max_longitude+0.00001)
    ax2.set_title('Zoomed-in')
    ax2.legend()
    ax2.grid()

    plt.show()

    return()


def read_data(file_path):
    '''
    Reads csv file as a pandas dataframe
    '''
    try:
        # Read the CSV file, skipping the second row (row index 1)
        df = pd.read_csv(file_path, skiprows=[1])
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        sys.exit(1)

    



def main(file_path='maracoos_01-20240124T1612_bffa_2334_762f.csv'):

    # Read the data file
    df = read_data(file_path)

    # Display some info of the DataFrame
    # print(df)

    # print("DataFrame Information:")
    # print(df.info())

    # # Print descriptive statistics
    # print("\nDescriptive Statistics:")
    # print(df.describe())

    # # Print the first few rows of the DataFrame
    # print("\nFirst Few Rows:")
    # print(df.head())


    # Store initial latitude and longitude
    prev_position = (df['lat_uv'].iloc[0], df['lon_uv'].iloc[0])

    # Initialize lists to store row numbers and corresponding unique latitude and longitude
    updated_rows = []
    actual_position_list = []
    commanded_position_list = []

    # Iterate through the DataFrame
    for index, row in df.iterrows():
        current_position = (row['lat_uv'], row['lon_uv'])
        commanded_position = (row['latitude'], row['longitude'])
        
        # Check if latitude and longitude have changed
        if prev_position is not None and current_position != prev_position:
            updated_rows.append(index)
            actual_position_list.append(current_position)
            commanded_position_list.append(commanded_position)
        
            # Plot one figure with two subplots 
            plot_positions(commanded_position, current_position, prev_position)

            # Update previous latitude and longitude
            prev_position = current_position


        # The loop runs only twice for this example
        if index > 100:
            break

if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        main()




