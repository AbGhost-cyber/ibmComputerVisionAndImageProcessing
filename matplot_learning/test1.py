import matplotlib.pyplot as plt

# fig, ax = plt.subplots()
# ax.plot([1, 2, 3, 4], [5, 4, 5, 4])
# plt.ylabel("Dummy")
# plt.xlabel("Duh")

fig = plt.figure() # an empty figure with no Axes
fig, ax = plt.subplots() # a figure with a single Axes
fig, axs = plt.subplots(2,2) # a figure with a 2X2 grid of Axes
# a figure with one axes

plt.show()

if __name__ == '__main__':
    print()
