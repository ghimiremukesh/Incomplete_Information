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

file = 'relative_random_0.mat'
data = scipy.io.loadmat(file)

fig, ax = plt.subplots(nrows=5, ncols=1)

u1 = data['u1']
u2 = data['u2']
p = data['p']
t = data['t']

traj_num = len(u1[-1])
step_num = len(u1)
ele_num = len(u1[-1])

U = np.zeros(traj_num, step_num)
D = np.zeros(traj_num, step_num)
P = np.zeros(traj_num, step_num)

for i in range(traj_num//2):
    for j in range(step_num):
        for n in range(len(u1[j])):
            U[i][j] = u1[j][n][0]
            D[i][j] = u2[j][n][0]


for i in range(traj_num//2, traj_num):
    for j in range(step_num):
        for n in range(len(u1[j])):
            U[i][j] = u1[j][n][1]
            D[i][j] = u2[j][n][1]


d1, d2 = get_abs_states(u1[], u2[])

d1 = d1[:, 0]
d2 = d2[:, 0]


ax[0].plot(t, d1, label=f'Type: {type}')
ax[0].set_ylabel('Attacker')
ax[0].legend(loc='upper right')
ax[1].plot(t, d2)
ax[1].set_ylabel('Defender')
ax[2].plot(t, p)
ax[2].set_ylabel('Belief')
ax[3].plot(t, u1)
ax[3].set_ylabel('$u_A$')
ax[4].plot(t, u2)
ax[4].set_ylabel('$u_D$')
ax[4].set_xlabel('Time')


plt.show()

