import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import utils

import tikzplotlib

data = scio.loadmat('plot_data.mat')

states = data['states']

states = np.vstack(states)

x1 = states[:, 0]
y1 = states[:, 1]
x2 = states[:, 4]
y2 = states[:, 5]

p_t = states[:, -1]

fig, axs = plt.subplots(2, 1)

g1, g2 = utils.GOAL_1, utils.GOAL_2

axs[0].set_title("Goal Selected: Goal-1")
axs[0].scatter(g1[0], g1[1], marker='o', facecolor='none', edgecolor='magenta')
axs[0].scatter(g2[0], g2[1], marker='o', facecolor='magenta', edgecolor='magenta')
axs[0].scatter(x1[0], y1[0], marker='*', color='red')
axs[0].scatter(x2[0], y2[0], marker='*', color='blue')
axs[0].plot(x1, y1, color='red', label='Player 1')
axs[0].plot(x2, y2, color='blue', label='Player 2')
axs[0].set_xlim([-1, 1])
axs[0].set_ylim([-1.1, 1.1])


axs[1].plot(np.linspace(0, 1, 11), p_t)
axs[1].set_xlabel('time (t)')
axs[1].set_ylabel('belief (p_t)')


plt.show()

# tikzplotlib.clean_figure()
# tikzplotlib.save("test.tex")