import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scipy.io
import matplotlib.pyplot as plt

file = 'hji_soccer_case_1.0.mat'
data = scipy.io.loadmat(file)

d1 = data['d1'].squeeze()
d2 = data['d2'].squeeze()
v1 = data['v1'].squeeze()
v2 = data['v2'].squeeze()
u1 = data['u1'].squeeze()
u2 = data['u2'].squeeze()
p = data['p'].squeeze()
V = data['V'].squeeze()
t = data['t'].squeeze()

fig, ax = plt.subplots(nrows=5, ncols=1)
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

fig2, ax2= plt.subplots(1, 1)
ax2.plot(d1, d2)

fig3, ax3 = plt.subplots(1, 1)
ax3.plot(t, V, label="NN Value")
ax3.set_ylabel('Value')
ax3.set_xlabel('TIme')

fig4, ax4 = plt.subplots(1, 1)
ax4.plot(t, p, label="NN Value")
ax4.set_ylabel('Belief')
ax4.set_xlabel('Time')

plt.show()