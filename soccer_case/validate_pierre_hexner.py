import numpy as np
from odeintw import odeintw
import scipy.io as scio
import matplotlib.pyplot as plt

import types

import utils


def dPdt(P, t, A, B, Q, R, S, ):
    n = A.shape[0]
    m = B.shape[1]

    if S is None:
        S = np.zeros((n, m))

    if isinstance(B, types.FunctionType):  # if B is time varying
        B_curr = B(t)
        B = B_curr

    return -(A.T @ P + P @ A - (P @ B + S) @ np.linalg.inv(R) @ (B.T @ P + S.T) + Q)


def dPhi(Phi, t, A):
    return np.dot(A, Phi)


def d(Phi, K, B, R, z):
    ds = np.zeros((len(Phi), 1))
    if isinstance(B, types.FunctionType):
        t_span = np.linspace(0, 1, 10)
        B_temp = np.array([B(i) for i in t_span])
    else:
        B_temp = np.array([B for _ in range(len(Phi))])

    B = B_temp
    for i in range(len(Phi)):
        # ds[i] = z.T @ Phi[i, :, :] @ K[i, :, :] @ B/R @ B.T @ K[i, :, :] @ Phi[i, :, :] @ z
        ds[i] = (z.T @ Phi[i, :, :].T @ K[i, :, :].T @ B[i] @ np.linalg.inv(R) @ B[i].T @ K[i, :, :] @ Phi[i, :, :] @ z)

    return ds


def value_hexner(x1, x2, p, t_step, Phi, K):
    """
    assuming R1 = R2 and A1 = A2, B1 = B2
    """
    z = np.array([[0], [1], [0], [0]])

    p1_val = p * (x1 - Phi[t_step, :, :] @ z).T @ K[t_step, :, :] @ (x1 - Phi[t_step, :, :] @ z) + \
             (1 - p) * (x1 + Phi[t_step, :, :] @ z).T @ K[t_step, :, :] @ (x1 + Phi[t_step, :, :] @ z)

    p2_val = p * (x2 - Phi[t_step, :, :] @ z).T @ K[t_step, :, :] @ (x2 - Phi[t_step, :, :] @ z) + \
             (1 - p) * (x2 + Phi[t_step, :, :] @ z).T @ K[t_step, :, :] @ (x2 + Phi[t_step, :, :] @ z)

    # value = np.sqrt(p1_val) - np.sqrt(p2_val)

    value = p1_val - p2_val

    return value


# define system
A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
# B1 = lambda t : np.array([[0], [t]])
B = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])
# B1 = np.array([[0],[1]])
Q = np.zeros((4, 4))
R1 = 0.01 * np.eye(2, 2)
# P1T = np.eye(2)
PT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
tspan = np.linspace(0, 1, 11)
tspan = np.flip(tspan)
K1 = odeintw(dPdt, PT, tspan, args=(A, B, Q, R1, None,))

K1 = np.flip(K1, axis=0)

A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
t_span = np.linspace(0, 1, 11)
t_span = np.flip(t_span)
PhiT = np.eye(4)

Phi_sol = odeintw(dPhi, PhiT, t_span, args=(A,))
Phi_sol = np.flip(Phi_sol, axis=0)

z = np.array([[0], [1], [0], [0]])
d1 = d(Phi_sol, K1, B, R1, z)
B2 = B
# B2 = lambda t : np.array([[0], [np.exp(-0.5*t)]])
R2 = 0.01 * np.eye(2)
K2 = odeintw(dPdt, PT, tspan, args=(A, B2, Q, R2, None,))
K2 = np.flip(K2, axis=0)
d2 = d(Phi_sol, K2, B2, R2, z)

## get states from file

data = scio.loadmat("states_pierre_square.mat")

states = data['states']
values = data['values']

val_hex = []
for i in range(len(states)):
    x1 = states[i, :4].reshape(-1, 1)
    x2 = states[i, 4:8].reshape(-1, 1)
    p = states[i, -1]
    val_hex.append(value_hexner(x1, x2, p, i, Phi_sol, K1).item())

plt.plot(np.linspace(0, 1, 11), values, label="value from pierre")
plt.plot(np.linspace(0, 1, 11), val_hex, label="value from hexner")
plt.legend()
plt.xlabel("t")
plt.ylabel("V")
plt.title("Value comparison")



# simulate hexner
x1 = np.array([[-1], [0], [0], [0]])
x2 = np.array([[-0.75], [0], [0], [0]])

states = []
states.append(np.hstack((x1.flatten(), x2.flatten())))
goal = 2
if goal == 1:
    p = 1
else:
    p = 0

ztheta = z * (2*p - 1)
curr_x = np.hstack((x1.flatten(), x2.flatten()))
U = []
D = []
val_hex_op = []
for i in range(10):
    x1 = curr_x[:4]
    x2 = curr_x[4:8]
    val_hex_op.append(value_hexner(x1.reshape(-1, 1), x2.reshape(-1, 1), p, i, Phi_sol, K1).item())
    u1 = utils.get_analytical_u(K1[i, :, :], R1, Phi_sol[i, :, :], x1.reshape(-1, 1), ztheta)
    u2 = utils.get_analytical_u(K1[i, :, :], R1, Phi_sol[i, :, :], x2.reshape(-1, 1), ztheta)
    U.append(u1.reshape(1, -1))
    D.append(u2.reshape(1, -1))
    curr_x = utils.go_forward(curr_x, u1.squeeze(), u2.squeeze(), 0.1)
    states.append(curr_x)

x1 = curr_x[:4]
x2 = curr_x[4:8]
val_hex_op.append(value_hexner(x1.reshape(-1, 1), x2.reshape(-1, 1), p, -1, Phi_sol, K1).item())

# plt.plot(np.linspace(0, 1, 11), val_hex_op)
# plt.xlabel("t")
# plt.ylabel("V")
# plt.title("Hexner's Value")

# plt.show()

states = np.vstack(states)

x1 = states[:, 0]
y1 = states[:, 1]
x2 = states[:, 4]
y2 = states[:, 5]

fig, axs = plt.subplots(2, 1)

g1, g2 = utils.GOAL_1, utils.GOAL_2
U = np.vstack(U)
D = np.vstack(D)
axs[1].plot(np.linspace(0, 0.9, 10), U[:, 0], label='$u_x$')
axs[1].plot(np.linspace(0, 0.9, 10), D[:, 0], label='$d_x$')
axs[1].plot(np.linspace(0, 0.9, 10), U[:, 1],  '-.', label='$u_y$')
axs[1].plot(np.linspace(0, 0.9, 10), D[:, 1], '--', label='$d_y$')
axs[1].set_xlim([-0.05, 1])
axs[1].legend()



if goal == 2:
    axs[0].scatter(g1[0], g1[1], marker='o', facecolor='none', edgecolor='magenta')
    axs[0].scatter(g2[0], g2[1], marker='o', facecolor='magenta', edgecolor='magenta')
else:
    axs[0].scatter(g1[0], g1[1], marker='o', facecolor='magenta', edgecolor='magenta')
    axs[0].scatter(g2[0], g2[1], marker='o', facecolor='none', edgecolor='magenta')

axs[0].annotate("1", (g1[0] + 0.01, g1[1]))
axs[0].annotate("2", (g2[0] + 0.01, g2[1]))

axs[0].scatter(x1[0], y1[0], marker='*', color='red')
axs[0].scatter(x2[0], y2[0], marker='*', color='blue')
axs[0].plot(x1, y1, color='red', label='A', marker='o', markersize=2)
axs[0].plot(x2, y2, color='blue', label='D', marker='o', markersize=2)
# axs[0].set_xlim([-1, 1])
# axs[0].set_ylim([-1, 1])
axs[0].legend()
axs[0].set_title("Hexner's Game")
plt.show()
