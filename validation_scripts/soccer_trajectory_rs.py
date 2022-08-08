import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scipy.io
import matplotlib.pyplot as plt


def get_abs_states(u1, u2):
    x_e = [np.array([0, 0])]  # both start at (0, 0) x_e is attacker
    x_p = [np.array([0, 0])]
    N = len(u1)
    dt = 1/N
    for i in range(N-1):
        x_e_new = np.array([x_e[i][0] + x_e[i][1] * dt,
                            x_e[i][1] + u1[i] * dt])

        x_p_new = np.array([x_p[i][0] + x_p[i][1] * dt,
                            x_p[i][1] + u2[i] * dt])
        x_e.append(x_e_new)
        x_p.append(x_p_new)

    x_e = np.asarray(x_e)
    x_p = np.asarray(x_p)

    return x_e, x_p

file = 'relative_0.0.mat'
data = scipy.io.loadmat(file)

d = data['d'].squeeze()
v = data['v'].squeeze()
u1 = data['u1'].squeeze()
u2 = data['u2'].squeeze()
p = data['p'].squeeze()
V = data['V'].squeeze()
t = data['t'].squeeze()


d1, d2 = get_abs_states(u1, u2)

d1 = d1[:, 0]
d2 = d2[:, 0]
fig, ax = plt.subplots(nrows=5, ncols=1, sharex=True)
ax[0].plot(t, d1)
ax[0].set_ylabel('Attacker')
ax[1].plot(t, d2)
ax[1].set_ylabel('Defender')
ax[2].plot(t, p)
ax[2].set_ylabel('Belief')
ax[3].plot(t, u1)
ax[3].set_ylabel('$u_A$')
ax[4].plot(t, u2)
ax[4].set_ylabel('$u_D$')
ax[4].set_xlabel('Time')

# fig2, ax2= plt.subplots(1, 1)
# ax2.plot(d1, d2)

fig3, ax3 = plt.subplots(1, 1)
ax3.plot(t, V, label="NN Value")
ax3.set_ylabel('Value')
ax3.set_xlabel('TIme')

fig4, ax4 = plt.subplots(1, 1)
ax4.plot(t, p)
ax4.set_ylabel('Belief')
ax4.set_xlabel('Time')

plt.show()

