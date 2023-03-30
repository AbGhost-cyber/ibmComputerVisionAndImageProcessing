import matplotlib.pyplot as plt
import numpy

# fig, ax = plt.subplots()
# ax.plot([1, 2, 3, 4], [5, 4, 5, 4])
# plt.ylabel("Dummy")
# plt.xlabel("Duh")

fig = plt.figure()  # an empty figure with no Axes
fig, ax = plt.subplots()  # a figure with a single Axes
fig, axs = plt.subplots(2, 2)  # a figure with a 2X2 grid of Axes
# a figure with one axes

# plt.show()

array1 = numpy.asarray([[[1, 2, 6], [2, 2, 3], [8, 3, 4]]])
array1[:, :, 2] = 0
if __name__ == '__main__':
    print(array1)
