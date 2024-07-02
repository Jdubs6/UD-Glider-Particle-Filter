import matplotlib.pyplot as plt
import random
import numpy as np
import csv
data = open(r'C:\Users\27imb\Downloads\maracoos_01-20240124T1612_bffa_2334_762f.csv')
csvFile= csv.reader(data)
ask_multiple = input("First run? Y or N")
initial_position_2 = []
commanded_position_2 = []
actual_position_2 = []
initial_position = []
commanded_position = []
actual_position = []
for num, row in enumerate(csvFile):
    if ask_multiple == "N":
        if num==76: #Selects row 77 in Excel
            initial_position_2.append(float(row[14])) #Selects the data from the cell in column 15 of the Excel sheet (So row 77, column 15). 
            initial_position_2.append(float(row[15]))
        if num==153: 
            commanded_position_2.append(float(row[4]))
            commanded_position_2.append(float(row[5]))
            actual_position_2.append(float(row[14]))
            actual_position_2.append(float(row[15]))
    if num==2:
        initial_position.append(float(row[14]))
        initial_position.append(float(row[15]))
    if num==71:
        commanded_position.append(float(row[4]))
        commanded_position.append(float(row[5]))
        actual_position.append(float(row[14]))
        actual_position.append(float(row[15]))
        continue
# Note: Some datasets do not have a commanded_position!!!!
number_of_samples = 200
#Time is about 2 hours
def calculate_weights(combined_positions,actual_position):
    position_weights = []
    for position in combined_positions:
        distance = np.sqrt((abs(position[0])-abs(actual_position[0]))**2+(abs(position[1])-abs(actual_position[1]))**2) #Distance between actual particle and estimated particle
        position_weight = np.exp(-distance / 0.0005) #0.0005 can be adjusted based on weight sensitivity. 
        position_weights.append(position_weight)
    position_weights = np.array(position_weights)
    position_weights /= np.sum(position_weights) #Normalizing the weights
    resampling = np.random.choice(range(number_of_samples),size=number_of_samples,replace=True,p=position_weights)
    resampled_positions = [combined_positions[idx] for idx in resampling]
    return(resampled_positions)
def estimate_positions(commanded_position, actual_position):
    combined_positions = []
    new_long = []
    new_lat = []
    estimated_positions_long = [commanded_position[0]] * number_of_samples
    estimated_positions_lat = [commanded_position[1]] * number_of_samples
    for position in estimated_positions_long:
        new_long.append(position + (1.5*random.uniform(-abs(abs(commanded_position[0])-abs(actual_position[0])),abs(abs(commanded_position[0])-abs(actual_position[0]))))) #Actual - Estimated
    for position in estimated_positions_lat:
        new_lat.append(position + (1.5*random.uniform(-abs(abs(commanded_position[1])-abs(actual_position[1])),abs(abs(commanded_position[1])-abs(actual_position[1]))))) #Actual - Estimated
    for a, b in zip(new_long,new_lat):
        combined_positions.append([a,b])
    return(calculate_weights(combined_positions,actual_position))
def estimate_positions_1(commanded_position_2):
    combined_positions = []
    new_long = []
    new_lat = []
    estimated_positions_long = [commanded_position_2[0]] * number_of_samples
    estimated_positions_lat = [commanded_position_2[1]] * number_of_samples
    for position in estimated_positions_long:
        new_long.append(position + (2*random.uniform(-abs(abs(commanded_position[0])-abs(actual_position[0])),abs(abs(commanded_position[0])-abs(actual_position[0]))))) #Actual - Estimated
    for position in estimated_positions_lat:
        new_lat.append(position + (2*random.uniform(-abs(abs(commanded_position[1])-abs(actual_position[1])),abs(abs(commanded_position[1])-abs(actual_position[1]))))) #Actual - Estimated
    for a, b in zip(new_long,new_lat):
        combined_positions.append([a,b])
    return(calculate_weights(combined_positions,commanded_position_2))
def plot_positions():
    if ask_multiple == "Y":
        combined_positions = estimate_positions(commanded_position,actual_position)
        plt.figure()
        for position in combined_positions:
            plt.scatter(position[0], position[1], color='blue', alpha=0.5)
        plt.scatter(commanded_position[0], commanded_position[1], color='green', marker='x', label='Commanded Position')
        plt.scatter(initial_position[0], initial_position[1], color='red', marker='x', label='Initial Position')
        plt.scatter(actual_position[0], actual_position[1], color='orange', marker='x', label='Actual Position')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Estimated Glider Position')
        plt.legend()
        plt.grid(True)
        plt.show()
    
        plt.figure()
        for position in combined_positions:
            plt.scatter(position[0], position[1], color='blue', alpha=0.5)
        plt.scatter(commanded_position[0], commanded_position[1], color='green', marker='x', label='Commanded Position')
        plt.scatter(actual_position[0], actual_position[1], color='orange', marker='x', label='Actual Position')
        plt.show()
    if ask_multiple == "N":
        combined_positions = estimate_positions_1(commanded_position_2)
        plt.figure()
        for position in combined_positions:
            plt.scatter(position[0], position[1], color='blue', alpha=0.5)
        plt.scatter(commanded_position_2[0], commanded_position_2[1], color='green', marker='x', label='Commanded Position')
        plt.scatter(initial_position_2[0], initial_position_2[1], color='red', marker='x', label='Initial Position')
        plt.scatter(actual_position_2[0], actual_position_2[1], color='orange', marker='x', label='Actual Position')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Estimated Glider Position')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        plt.figure()
        for position in combined_positions:
            plt.scatter(position[0], position[1], color='blue', alpha=0.5)
        plt.scatter(commanded_position_2[0], commanded_position_2[1], color='green', marker='x', label='Commanded Position')
        plt.scatter(actual_position_2[0], actual_position_2[1], color='orange', marker='x', label='Actual Position')
        plt.show()
    else:
        return()
plot_positions()
