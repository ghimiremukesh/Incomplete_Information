import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scipy.io
import matplotlib.pyplot as plt


def get_abs_states(u1, u2):
    x_e = [np.array([0.1, 0])]  # both start at (0, 0) x_e is attacker
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
fig1, ax1 = plt.subplots(nrows=5, ncols=1, sharex=True) # for right
fig2, ax2 = plt.subplots(nrows=5, ncols=1, sharex=True) # for left

count_1 = 0
count_2 = 0
for i in range(1):
    file = f'discrete_value_{i}.mat'
    # file = f'relative_0.2.mat'
    data = scipy.io.loadmat(file)

    d = data['d'].squeeze()
    v = data['v'].squeeze()
    u1 = data['u1'].squeeze()
    u2 = data['u2'].squeeze()
    p = data['p'].squeeze()
    V = data['V'].squeeze()
    t = data['t'].squeeze()
    type = data['type'].squeeze()
    p_0 = data['p_0'].squeeze()
    # if (type == 1 and count_1 == 0) or (type == 0 and count_2 == 0):
    #     count_1 = 1 if type == 1 else count_1
    #     count_2 = 1 if type == 0 else count_2

    print(type)
    print(data['p_0'].squeeze())
    d1, d2 = get_abs_states(u1, u2)

    d1 = d1[:, 0]
    d2 = d2[:, 0]
    if type == 0:
        ax1[0].plot(t, d1, label=f'{type}')
        ax1[0].set_ylabel('Attacker')
        # ax1[0].legend(loc='upper right', bbox_to_anchor=(1.13, 1.1))
        ax1[1].plot(t, d2)
        ax1[1].set_ylabel('Defender')
        ax1[2].plot(t, p)
        ax1[2].set_ylabel('Belief')
        ax1[3].plot(t, u1)
        ax1[3].set_ylabel('$u_A$')
        ax1[4].plot(t, u2)
        ax1[4].set_ylabel('$u_D$')
        ax1[4].set_xlabel('Time')
        ax1[0].set_title('Trajectory for Type Right')
    else:
        ax2[0].plot(t, d1, label=f'{type}')
        ax2[0].set_ylabel('Attacker')
        # ax2[0].legend(loc='upper right', bbox_to_anchor=(1.13, 1.1))
        ax2[1].plot(t, d2)
        ax2[1].set_ylabel('Defender')
        ax2[2].plot(t, p)
        ax2[2].set_ylabel('Belief')
        ax2[3].plot(t, u1)
        ax2[3].set_ylabel('$u_A$')
        ax2[4].plot(t, u2)
        ax2[4].set_ylabel('$u_D$')
        ax2[4].set_xlabel('Time')
        ax2[0].set_title('Trajectory for Type Left')


    # fig2, ax2= plt.subplots(1, 1)
    # ax2.plot(d1, d2)

    # fig3, ax3 = plt.subplots(1, 1)
    # ax3.plot(t, V, label="NN Value")
    # ax3.set_ylabel('Value')
    # ax3.set_xlabel('TIme')
    #
    # fig4, ax4 = plt.subplots(1, 1)
    # ax4.plot(t, p)
    # ax4.set_ylabel('Belief')
    # ax4.set_xlabel('Time')
plt.show()

