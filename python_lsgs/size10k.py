import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from lsgs_solver import lsgssolver  # Assuming you have implemented the lsgs solver in a separate module

# Load the 10000 gate circuit data
print("\nThis script loads a 10000 gate example circuit and")
print("\nuses lsgs to size it for 5 timing specifications.")
print("\nEach sizing should take around 10-20 seconds.")

data = loadmat('ckt10k.mat')  # Load the data for the 10000 gate circuit
print("\nExample circuit with 10000 gates loaded.")

# Extract the data from the .mat file
a = data['a']  # Example gate sizes
g = data['g']  # Gate parameters
F = data['F']  # Some function or matrix related to the circuit
dmin = data['dmin']  # Minimum delay constraint

T = [100, 110, 120, 130, 140]  # Set of timing specifications

print("\nSizing 10000 gate circuit:")
area = []

# Loop through the timing specifications
for timing_spec in T:
    print(f'\tfor timing specification T = {timing_spec:.3f} ...', end='')
    x = lsgssolver(a.shape[0], a, g, F, F.T, dmin, timing_spec, [], None, 250, None, None, None, None, None, 250)
    area.append(np.dot(a.T, x))
    print(f'done. Area = {area[-1]:.3e}')

print('Done.')

# Plot the trade-off curve
plt.plot(T, area)
plt.xlabel('Timing specification')
plt.ylabel('Area')
plt.title('Trade-off curve')
plt.show()

