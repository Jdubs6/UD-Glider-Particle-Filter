import numpy as np
import matplotlib.pyplot as plt



def update_weights(particles, measurement, measurement_std):
    """ Update particle weights based on the measurement. """
    distances = np.linalg.norm(particles - measurement, axis=1)
    weights = np.exp(-0.5 * (distances / measurement_std) ** 2)
    return weights / np.sum(weights)  # Normalize weights

def resample(particles, measurement, measurement_std=0.001):
    """ Resample particles based on weights. """
    weights = update_weights(particles, measurement, measurement_std)
    indices = np.random.choice(len(particles), len(particles), p=weights)
    return particles[indices]



# Example usage
# print(np.shape(particles))
# print(np.shape(resampled_particles))
# plt.scatter(particles[:,0],particles[:,1], marker='x')
# plt.scatter(measurement[0],measurement[1], color='red', marker='+')

# plt.scatter(resampled_particles[:,0],resampled_particles[:,1], marker='o')
# plt.show()
# print(particles)
