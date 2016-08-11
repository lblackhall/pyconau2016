import numpy as np
from pylab import ylim, title, ylabel, xlabel
import matplotlib.pyplot as plt
from kalman import SingleStateKalmanFilter
from moving_average import MovingAverageFilter

# Create some random temperature data
random_data_init = np.random.normal(20, 1, size=(1, 100)).flatten()
random_data_then = np.random.normal(22, 1, size=(1, 100)).flatten()
random_data = np.hstack((random_data_init, random_data_then))

# Initialise the Kalman Filter

A = 1  # No process innovation
C = 1  # Measurement
B = 0  # No control input
Q = 0.005  # Process covariance
R = 1  # Measurement covariance
x = 18  # Initial estimate
P = 1  # Initial covariance

kalman_filter = SingleStateKalmanFilter(A, B, C, x, P, Q, R)

# Initialise two moving average filters with different window lengths
ma5_filter = MovingAverageFilter(5)
ma50_filter = MovingAverageFilter(50)

# Empty lists for capturing filter estimates
kalman_filter_estimates = []
ma5_filter_estimates = []
ma50_filter_estimates = []

# Simulate the data arriving sequentially
for data in random_data:
    ma5_filter.step(data)
    ma5_filter_estimates.append(ma5_filter.current_state())

    ma50_filter.step(data)
    ma50_filter_estimates.append(ma50_filter.current_state())

    kalman_filter.step(0, data)
    kalman_filter_estimates.append(kalman_filter.current_state())


# Plot the Data for Presentation
plt.plot(random_data, 'r*')
title("Filtering Real-Time Data")
ylabel('Temperature')
xlabel('Sample')
ylim([15, 25])
plt.plot(ma5_filter_estimates, 'b', linewidth=2.0)
plt.plot(ma50_filter_estimates, 'g', linewidth=2.0)
plt.plot(kalman_filter_estimates, 'k', linewidth=2.0)

# Show the plot
plt.show()