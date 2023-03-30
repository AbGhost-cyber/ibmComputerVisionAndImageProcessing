import matplotlib.pyplot as plt
import numpy as np

# Using NumPy's np.linspace() function, an array of 200 evenly spaced values
# is created in the range from 0 to 2π (2 * np.pi) [1].
# These values represent the x-coordinates for the sine wave.
x = np.linspace(0, 2 * np.pi, 200)

# The np.sin() function calculates the sine of each value in the x array,
# resulting in an array of corresponding y-coordinates for the sine wave
y = np.sin(x)

# fig represents the entire figure, while ax refers to the specific axis
# within the figure on which the plot will be displayed
fig, ax = plt.subplots()

# The ax.plot() function is used to create a line plot of the sine wave,
# taking x as the x-coordinates and y as the y-coordinates
ax.plot(x, y)

plt.show()

if __name__ == '__main__':
    print(x)
    print(y)
