Hey there, hope you're having a great day so far! This repository contains the progress on the particle filter used for the glider project in the Autonomous Underwater Systems Bootcamp. 

Contained are two files: "Particle Filter Part 2 with Weights -- with File Opening" and "Particle Filter Part 2 with Weights"

The filters can take up to two sets of datapoints at a time, each having an initial_position, commanded_position, and actual_position. 
When the code is ran, it will ask if this the "First run? Y or N". Noise is added to the dataset based on how close the actual position of the glider is to the commanded position of the glider. 

Typing "Y" and then "Enter" will take only the first set of datapoints (initial_position, commanded_position, and actual_position) and create noise based on the positions of the commanded_position and actual_position. They are also multiplied by a value (1.5) to get a more accurate range of values for where the glider could be. The weight system is then used to narrow down the data points and two particle filter graphs are produced, one with the initial position and one zoomed in on the commanded and actual positions of the glider. The actual_position is needed for this trial. 

Typing "N" and "Enter" will take both sets of glider position datapoints. Using the noise from the first data set, the filter will attempt to predict, based on "commanded_position_2", where the glider will actually surface in it's next cycle. The "actual_position_2" datapoint is only used to see how accurate the filter was in predicting the location of the glider, and is not used in calculations. The outcome of the particle filter will then be graphed. 

The only major difference between the two particle filters attached to this repository is that one uses variables to store the glider positioning data, while the other can open a CSV file directly from your computer and analyze the data from it. I have attached the CSV file used in the example calculations. 
IMPORTANT NOTE: You will have to change the path located in the variable "data" in order to properly open the csv file, once it's been downloaded onto your computer. 

This code is a work in progress, and thus could be very ugly. I also am quite new to particle filters, so any edits, suggestions, optimizations, or advice would be much appreciated! My email is jaredwie@udel.edu if you have questions! :)
