import pickle
import numpy as np
from matplotlib import pyplot as plt

def plot_curve(ax, x, y, x_label, y_label, title, rotate=False):
    ax.plot(x, y)

    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.tick_params(axis='x', rotation=90)

max_sims = None

with open('h_max_sims_over_image_space.data', 'rb') as f:
    max_sims = pickle.load(f)

h_means = None

with open('h_means_over_image_space.data', 'rb') as f:
    h_means = pickle.load(f)

with open('ignored.data', 'rb') as f:
    ignored = pickle.load(f)

fig, axes = plt.subplots(1, 2)

plot_curve(axes[0], np.arange(0, len(h_means)), h_means, 
           'Similarity mean', 'Image id', 'Harmonic mean of similarity means')

plot_curve(axes[1], np.arange(0, len(max_sims)), max_sims,
           'Max', 'Image id', 'Harmonic mean of max similiarity')

fig.tight_layout()

print(ignored)

plt.show()