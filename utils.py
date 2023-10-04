# utility functions
import numpy as np
import os
from itertools import product
from odeintw import odeintw
import torch


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def final_value(x1, v1, x2, v2, p):
    return p * ((x1 - 1) ** 2 - (x2 - 1) ** 2) + (1 - p) * ((x1 + 1) ** 2 - (x2 + 1) ** 2) + v1 ** 2 - v2 ** 2


def running_payoff(u, d, R1, R2):
    payoff = (R1 * u ** 2 - R2 * d ** 2)

    return np.array(payoff)


def final_value_reformulation(x, u, d, Phi, K1, K2):
    # assume tau = 0.1
    tau = 0.1

    # here x is x_prev
    p = x[..., -1].reshape(-1, 1)  # belief
    x = x[..., 1:-1]  # state 1:-1 or 0:-1 depending on if time is included
    x1 = x[..., :2]  # player 1's state
    x2 = x[..., 2:]  # player 2's state

    B = np.array([[0], [1]])

    R1 = 1
    R2 = 5

    ztheta = np.concatenate((np.ones_like(p), np.zeros_like(p)), axis=1).T

    # instantaneous loss term
    term1 = p * ((x1.T - (Phi @ ztheta)).T.dot(K1) * ((x1.T - (Phi @ ztheta)).T)).sum(axis=1).reshape(-1, 1) + \
            (1 - p) * ((x1.T + (Phi @ ztheta)).T.dot(K1) * ((x1.T + (Phi @ ztheta)).T)).sum(axis=1).reshape(-1, 1)

    term2 = p * ((x2.T - (Phi @ ztheta)).T.dot(K2) * ((x2.T - (Phi @ ztheta)).T)).sum(axis=1).reshape(-1, 1) + \
            (1 - p) * ((x2.T + (Phi @ ztheta)).T.dot(K2) * ((x2.T + (Phi @ ztheta)).T)).sum(axis=1).reshape(-1, 1)

    int_t1 = p * (1 / R1) * ((x1.T - (Phi @ ztheta)).T.dot(K1.T @ B @ B.T @ K1) * \
                             ((x1.T - (Phi @ ztheta)).T)).sum(axis=1).reshape(-1, 1) + (1 - p) * \
             (1 / R1) * ((x1.T + (Phi @ ztheta)).T.dot(K1.T @ B @ B.T @ K1) * \
                         ((x1.T + (Phi @ ztheta)).T)).sum(axis=1).reshape(-1, 1)

    int_t2 = p * (1 / R2) * ((x2.T - (Phi @ ztheta)).T.dot(K2.T @ B @ B.T @ K2) * \
                             ((x2.T - (Phi @ ztheta)).T)).sum(axis=1).reshape(-1, 1) + (1 - p) * \
             (1 / R2) * ((x2.T + (Phi @ ztheta)).T.dot(K2.T @ B @ B.T @ K2) * \
                         ((x2.T + (Phi @ ztheta)).T)).sum(axis=1).reshape(-1, 1)

    action_term1 = 2 * p * u.reshape(-1, 1) * (B.T @ K1 @ (x1.T - (Phi @ ztheta))).T + \
                   2 * (1 - p) * u.reshape(-1, 1) * (B.T @ K1 @ (x1.T + (Phi @ ztheta))).T

    action_term2 = 2 * p * d.reshape(-1, 1) * (B.T @ K2 @ (x2.T - (Phi @ ztheta))).T + \
                   2 * (1 - p) * d.reshape(-1, 1) * (B.T @ K2 @ (x2.T + (Phi @ ztheta))).T

    return tau * (int_t1 - int_t2) + (term1 - term2) + tau * (action_term1 - action_term2)



def get_running_payoff(u, d, R1, R2, tau=0.1):
    u = np.array(u)
    d = np.array(d)
    u_d = list(product(u, d))
    payoff = [(R1 * u ** 2 - R2 * d ** 2) for u, d in u_d]
    return np.array(payoff)


def convex_hull(points, vex=True):
    """Computes the convex hull of a set of 2D points.

    Input: an iterable sequence of (x, y) pairs representing the points.
    Output: a list of vertices of the convex hull in counter-clockwise order,
      starting from the vertex with the lexicographically smallest coordinates.
    Implements Andrew's monotone chain algorithm. O(n log n) complexity.
    """
    # Sort the points lexicographically (tuples are compared lexicographically).
    # Remove duplicates to detect the case we have just one unique point.
    points = sorted(set(points))

    # Boring case: no points or a single point, possibly repeated multiple times.
    if len(points) <= 1:
        return points

    # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
    # Returns a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    if vex:
        # Build lower hull
        lower = []
        for p in points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
    else:
        # Build upper hull
        upper = []
        for p in reversed(points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

    # Concatenation of the lower and upper hulls gives the convex hull.
    # Last point of each list is omitted because it is repeated at the beginning of the other list.
    return lower[:] if vex else upper[:]


def cav_vex(values, p, type='vex', num_ps=11):
    lower = True if type == 'vex' else False
    ps = np.linspace(0, 1, num_ps)
    points = zip(ps, values)
    hull_points = convex_hull(points, vex=lower)
    hull_points = sorted(hull_points)
    x, y = zip(*hull_points)
    num_facets = len(hull_points) - 1
    if p != 1:
        s_idx = [True if x[i] <= p < x[i + 1] else False for i in range(num_facets)]
    else:
        s_idx = [True if x[i] < p <= x[i + 1] else False for i in range(num_facets)]
    assert sum(s_idx) == 1, "p must belong to only one interval, check for bugs!"
    facets = np.array(list(zip(x, x[1:])))
    val_zips = np.array(list(zip(y, y[1:])))
    P = facets[s_idx].flatten()
    vals = val_zips[s_idx].flatten()
    x1, x2 = P
    y1, y2 = vals
    # calculate the value from the equation:
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope * p + intercept


def make_payoff(x, y):
    x_minus = x[1]
    x_plus = y[1]
    xv_minus = x[2]
    xv_plus = y[2]
    y_minus = x[3]
    y_plus = y[3]
    yv_minus = x[4]
    yv_plus = y[4]

    X_minus = (x_minus, xv_minus)
    X_plus = (x_plus, xv_plus)
    Y_minus = (y_minus, yv_minus)
    Y_plus = (y_plus, yv_plus)

    X = [X_minus, X_plus]
    Y = [Y_minus, Y_plus]

    pairs = np.array(list(product(X, Y)))

    X_return = pairs.reshape(-1, 4)
    X_return = np.hstack((x[0] * np.ones((4, 1)), X_return))

    return X_return


def initial_payoff_term(x, K1, K2, Phi):
    """

    :param x: current state
    :param K1: current K matrix
    :param Phi: current Phi matrix
    :return: "initial payoff"
    """
    p = x[..., -1].reshape(-1, 1)  # belief
    x = x[..., 1:-1]  # state 1:-1 or 0:-1 depending on if time is included
    x1 = x[..., :2]  # player 1's state
    x2 = x[..., 2:]  # player 2's state

    ztheta = np.concatenate((np.ones_like(p), np.zeros_like(p)), axis=1).T

    term1 = p * ((x1.T - (Phi @ ztheta)).T.dot(K1) * ((x1.T - (Phi @ ztheta)).T)).sum(axis=1).reshape(-1, 1) + \
            (1 - p) * ((x1.T + (Phi @ ztheta)).T.dot(K1) * ((x1.T + (Phi @ ztheta)).T)).sum(axis=1).reshape(-1, 1)

    # term1 = p * np.sum(K1 @ np.square(x1.T - (Phi @ ztheta)), axis=0).reshape(-1, 1) + \
    #         (1 - p) * np.sum(K1 @ np.square(x1.T + (Phi @ ztheta)), axis=0).reshape(-1, 1)

    term2 = p * ((x2.T - (Phi @ ztheta)).T.dot(K2) * ((x2.T - (Phi @ ztheta)).T)).sum(axis=1).reshape(-1, 1) + \
            (1 - p) * ((x2.T + (Phi @ ztheta)).T.dot(K2) * ((x2.T + (Phi @ ztheta)).T)).sum(axis=1).reshape(-1, 1)

    return term1 - term2


def dynamics(x, u, d, DT):
    def get_p2_a(x, d_max):
        v = x[..., 4]  # velocity
        # a_l = -d_max * (1 - np.maximum(np.zeros_like(v), 0.5 * np.sign(v)))
        # a_r = d_max * (1 - np.maximum(np.zeros_like(v), 0.5 * np.sign(-v)))
        a_l = -d_max * np.ones_like(v)
        a_r = d_max * np.ones_like(v)

        return a_l, a_r

    U = np.array([-np.ones_like(x[..., 2]).reshape(-1, 1), np.ones_like(x[..., 2]).reshape(-1, 1)]).T.squeeze()
    d1, d2 = get_p2_a(x, d)
    D = np.vstack((d1, d2)).T
    dt = DT
    x1_new = x[..., 1].reshape(-1, 1) + x[..., 2].reshape(-1, 1) * dt + 0.5 * U * dt ** 2
    x2_new = x[..., 3].reshape(-1, 1) + x[..., 4].reshape(-1, 1) * dt + 0.5 * D * dt ** 2
    v1_new = x[..., 2].reshape(-1, 1) + U * dt
    v2_new = x[..., 4].reshape(-1, 1) + D * dt

    # v_new = np.array([x[..., 2] + a for a in da_v]).T.reshape(-1, 1)
    # p_new = np.multiply(x[:, 3].reshape(-1, 1),
    #                     np.ones((x.shape[0], 4))).reshape(-1, 1)
    X_new = np.hstack((x1_new.reshape(-1, 1), v1_new.reshape(-1, 1), x2_new.reshape(-1, 1),
                       v2_new.reshape(-1, 1)))  # returns new states, n x 8

    return X_new


def stage_cost(x, K1, K2, R1, R2, Phi, u, d):
    """

    :param x: current state
    :param K1: current K for p1
    :param K2: current K for p2
    :param R1: R for p1
    :param R2: R for p2
    :param Phi: current Phi
    :param u: p1 control
    :param d: p2 control
    :return:
    """
    dt = 0.1
    B = np.array([[0], [1]])
    p = x[..., -1].reshape(-1, 1)
    x = x[..., 1:-1]
    x1 = x[..., :2]
    x2 = x[..., 2:]

    ztheta = np.concatenate((np.ones_like(p), np.zeros_like(p)), axis=1).T

    term1 = p.reshape(-1, 1) * ((R1 * u + B.T @ K1 @ (x1.T - (Phi @ ztheta))) ** 2).reshape(-1, 1) * (1 / R1) + \
            (1 - p).reshape(-1, 1) * ((R1 * u + B.T @ K1 @ (x1.T + (Phi @ ztheta))) ** 2).reshape(-1, 1) * (1 / R1)

    term2 = p.reshape(-1, 1) * ((R2 * d + B.T @ K2 @ (x2.T - (Phi @ ztheta))) ** 2).reshape(-1, 1) * (1 / R2) + \
            (1 - p).reshape(-1, 1) * ((R2 * d + B.T @ K2 @ (x2.T + (Phi @ ztheta))) ** 2).reshape(-1, 1) * (1 / R2)

    return dt * (term1 - term2)  # integral approximation


def initial_payoff_term(x, K1, K2, Phi):
    """

    :param x: current state
    :param K1: current K matrix
    :param Phi: current Phi matrix
    :return: "initial payoff"
    """
    p = x[..., -1].reshape(-1, 1)  # belief
    x = x[..., 1:-1]  # state 1:-1 or 0:-1 depending on if time is included
    x1 = x[..., :2]  # player 1's state
    x2 = x[..., 2:]  # player 2's state

    ztheta = np.concatenate((np.ones_like(p), np.zeros_like(p)), axis=1).T

    term1 = p * ((x1.T - (Phi @ ztheta)).T.dot(K1) * ((x1.T - (Phi @ ztheta)).T)).sum(axis=1).reshape(-1, 1) + \
            (1 - p) * ((x1.T + (Phi @ ztheta)).T.dot(K1) * ((x1.T + (Phi @ ztheta)).T)).sum(axis=1).reshape(-1, 1)

    # term1 = p * np.sum(K1 @ np.square(x1.T - (Phi @ ztheta)), axis=0).reshape(-1, 1) + \
    #         (1 - p) * np.sum(K1 @ np.square(x1.T + (Phi @ ztheta)), axis=0).reshape(-1, 1)

    term2 = p * ((x2.T - (Phi @ ztheta)).T.dot(K2) * ((x2.T - (Phi @ ztheta)).T)).sum(axis=1).reshape(-1, 1) + \
            (1 - p) * ((x2.T + (Phi @ ztheta)).T.dot(K2) * ((x2.T + (Phi @ ztheta)).T)).sum(axis=1).reshape(-1, 1)

    return term1 - term2


def make_payoff_3(x, y, z):
    x_minus = x[1]
    x_zero = y[1]
    x_plus = z[1]
    xv_minus = x[2]
    xv_zero = y[2]
    xv_plus = z[2]
    y_minus = x[3]
    y_zero = y[3]
    y_plus = z[3]
    yv_minus = x[4]
    yv_zero = y[4]
    yv_plus = z[4]

    X_minus = (x_minus, xv_minus)
    X_zero = (x_zero, xv_zero)
    X_plus = (x_plus, xv_plus)
    Y_minus = (y_minus, yv_minus)
    Y_zero = (y_zero, yv_zero)
    Y_plus = (y_plus, yv_plus)

    X = [X_minus, X_zero, X_plus]
    Y = [Y_minus, Y_zero, Y_plus]

    pairs = np.array(list(product(X, Y)))

    X_return = pairs.reshape(-1, 4)
    X_return = np.hstack((x[0] * np.ones((9, 1)), X_return))

    return X_return


def dynamics_3(x, u, d, DT):
    def get_p2_a(x, d_max):
        v = x[..., 4]  # velocity
        # a_l = -d_max * (1 - np.maximum(np.zeros_like(v), 0.5 * np.sign(v)))
        # a_r = d_max * (1 - np.maximum(np.zeros_like(v), 0.5 * np.sign(-v)))
        a_l = -d_max * np.ones_like(v)
        a_r = d_max * np.ones_like(v)

        return a_l, a_r

    # U = np.array([-np.ones_like(x[..., 2]).reshape(-1, 1), np.ones_like(x[..., 2]).reshape(-1, 1)]).T.squeeze()
    U = np.array([-np.ones_like(x[..., 2]).reshape(-1, 1), np.zeros_like(x[..., 2]).reshape(-1, 1),
                  np.ones_like(x[..., 2]).reshape(-1, 1)]).T.squeeze()
    d1, d2 = get_p2_a(x, d)
    D = np.vstack((d1, np.zeros_like(d1), d2)).T  # add zero to action
    dt = DT
    x1_new = x[..., 1].reshape(-1, 1) + x[..., 2].reshape(-1, 1) * dt + 0.5 * U * dt ** 2
    x2_new = x[..., 3].reshape(-1, 1) + x[..., 4].reshape(-1, 1) * dt + 0.5 * D * dt ** 2
    v1_new = x[..., 2].reshape(-1, 1) + U * dt
    v2_new = x[..., 4].reshape(-1, 1) + D * dt

    # v_new = np.array([x[..., 2] + a for a in da_v]).T.reshape(-1, 1)
    # p_new = np.multiply(x[:, 3].reshape(-1, 1),
    #                     np.ones((x.shape[0], 4))).reshape(-1, 1)
    X_new = np.hstack((x1_new.reshape(-1, 1), v1_new.reshape(-1, 1), x2_new.reshape(-1, 1),
                       v2_new.reshape(-1, 1)))  # returns new states, n x 8

    return X_new


def go_forward(x, u, d, DT):
    pos1 = x[1]
    vel1 = x[2]
    pos2 = x[3]
    vel2 = x[4]

    pos1 += u * DT + 0.5 * u * DT ** 2
    vel1 += u * DT
    pos2 += d * DT + 0.5 * d * DT ** 2
    vel2 += d * DT

    return np.array([pos1, vel1, pos2, vel2])


def solve_hexner(R):
    # solve ricatti ode
    def dPdt(P, t, A, B, Q, R, S, ):
        if S is None:
            S = np.zeros((A.shape[0], B.shape[1]))

        return -(A.T @ P + P @ A - (P @ B + S) / R @ (B.T @ P + S.T) + Q)

    # solve Phi
    def dPhi(Phi, t, A):
        return np.dot(A, Phi)

    # solve for K
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    Q = np.zeros((2, 2))
    R = R
    PT = np.eye(2)
    # PT = np.array([[1, 0], [0, 0]])
    tspan = np.linspace(0, 1, 11)
    tspan = np.flip(tspan)
    K = odeintw(dPdt, PT, tspan, args=(A, B, Q, R, None,))

    K = np.flip(K, axis=0)  # flip to forward time

    # solve for phi
    PhiT = np.eye(2)

    Phi_sol = odeintw(dPhi, PhiT, tspan, args=(A,))
    Phi_sol = np.flip(Phi_sol, axis=0)

    return K[:-1], Phi_sol[:-1]  # no need the last one


def get_analytical_u(K, R, Phi, x, ztheta):
    B = np.array([[0], [1]])
    u = -(1 / R) * B.T @ K @ x + (1 / R) * B.T @ K @ Phi @ (ztheta)

    return u


def compute_d(Phi, K, R):
    B = np.array([[0.], [1.]])
    z = np.array([[1.], [0.]])

    ds = np.zeros((len(Phi), 1))
    for i in range(len(Phi)):
        ds[i] = z.T @ Phi[i, :, :] @ K[i, :, :] @ B / R @ B.T @ K[i, :, :] @ Phi[i, :, :] @ z

    return ds


def make_payoff_5(x, y, z, a, b):
    x_minus = x[1]
    x_minus_five = y[1]
    x_zero = z[1]
    x_plus_five = a[1]
    x_plus = b[1]

    xv_minus = x[2]
    xv_minus_five = y[2]
    xv_zero = z[2]
    xv_plus_five = a[2]
    xv_plus = b[2]

    y_minus = x[3]
    y_minus_five = y[3]
    y_zero = z[3]
    y_plus_five = a[3]
    y_plus = b[3]

    yv_minus = x[4]
    yv_minus_five = y[4]
    yv_zero = z[4]
    yv_plus_five = a[4]
    yv_plus = b[4]

    X_minus = (x_minus, xv_minus)
    X_minus_five = (x_minus_five, xv_minus_five)
    X_zero = (x_zero, xv_zero)
    X_plus_five = (x_plus_five, xv_plus_five)
    X_plus = (x_plus, xv_plus)
    Y_minus = (y_minus, yv_minus)
    Y_minus_five = (y_minus_five, yv_minus_five)
    Y_zero = (y_zero, yv_zero)
    Y_plus_five = (y_plus_five, yv_plus_five)
    Y_plus = (y_plus, yv_plus)

    X = [X_minus, X_minus_five, X_zero, X_plus_five, X_plus]
    Y = [Y_minus, Y_minus_five, Y_zero, Y_plus_five, Y_plus]

    pairs = np.array(list(product(X, Y)))

    X_return = pairs.reshape(-1, 4)
    X_return = np.hstack((x[0] * np.ones((25, 1)), X_return))

    return X_return


def dynamics_5(x, u, d, DT):
    # U = np.array([-np.ones_like(x[..., 2]).reshape(-1, 1), np.ones_like(x[..., 2]).reshape(-1, 1)]).T.squeeze()
    # U = np.array([-np.ones_like(x[..., 2]).reshape(-1, 1), np.zeros_like(x[..., 2]).reshape(-1, 1),
    #               np.ones_like(x[..., 2]).reshape(-1, 1)]).T.squeeze()

    U = np.array([-np.ones_like(x[..., 2]).reshape(-1, 1), -0.5 * np.ones_like(x[..., 2]).reshape(-1, 1),
                  0 * np.ones_like(x[..., 2]).reshape(-1, 1), 0.5 * np.ones_like(x[..., 2]).reshape(-1, 1),
                  1 * np.ones_like(x[..., 2]).reshape(-1, 1)]).T.squeeze()

    D = U
    # d1, d2 = get_p2_a(x, d)
    # D = np.vstack((d1, np.zeros_like(d1), d2)).T   # add zero to action
    dt = DT
    x1_new = x[..., 1].reshape(-1, 1) + x[..., 2].reshape(-1, 1) * dt + 0.5 * U * dt ** 2
    x2_new = x[..., 3].reshape(-1, 1) + x[..., 4].reshape(-1, 1) * dt + 0.5 * D * dt ** 2
    v1_new = x[..., 2].reshape(-1, 1) + U * dt
    v2_new = x[..., 4].reshape(-1, 1) + D * dt

    # v_new = np.array([x[..., 2] + a for a in da_v]).T.reshape(-1, 1)
    # p_new = np.multiply(x[:, 3].reshape(-1, 1),
    #                     np.ones((x.shape[0], 4))).reshape(-1, 1)
    X_new = np.hstack((x1_new.reshape(-1, 1), v1_new.reshape(-1, 1), x2_new.reshape(-1, 1),
                       v2_new.reshape(-1, 1)))  # returns new states, n x 8

    return X_new


def dynamics_n(x, num_actions, DT):
    # U = np.array([-np.ones_like(x[..., 2]).reshape(-1, 1), np.ones_like(x[..., 2]).reshape(-1, 1)]).T.squeeze()
    # U = np.array([-np.ones_like(x[..., 2]).reshape(-1, 1), np.zeros_like(x[..., 2]).reshape(-1, 1),
    #               np.ones_like(x[..., 2]).reshape(-1, 1)]).T.squeeze()
    us = torch.linspace(-1, 1, num_actions)

    # U = np.array([-np.ones_like(x[..., 2]).reshape(-1, 1), -0.5 * np.ones_like(x[..., 2]).reshape(-1, 1),
    #               0 * np.ones_like(x[..., 2]).reshape(-1, 1), 0.5 * np.ones_like(x[..., 2]).reshape(-1, 1),
    #               1 * np.ones_like(x[..., 2]).reshape(-1, 1)]).T.squeeze()

    U = us.repeat(x.shape[0], 1).reshape(-1, num_actions).numpy()

    D = U
    # d1, d2 = get_p2_a(x, d)
    # D = np.vstack((d1, np.zeros_like(d1), d2)).T   # add zero to action
    dt = DT
    x1_new = x[..., 1].reshape(-1, 1) + x[..., 2].reshape(-1, 1) * dt + 0.5 * U * dt ** 2
    x2_new = x[..., 3].reshape(-1, 1) + x[..., 4].reshape(-1, 1) * dt + 0.5 * D * dt ** 2
    v1_new = x[..., 2].reshape(-1, 1) + U * dt
    v2_new = x[..., 4].reshape(-1, 1) + D * dt

    # v_new = np.array([x[..., 2] + a for a in da_v]).T.reshape(-1, 1)
    # p_new = np.multiply(x[:, 3].reshape(-1, 1),
    #                     np.ones((x.shape[0], 4))).reshape(-1, 1)
    X_new = np.hstack((x1_new.reshape(-1, 1), v1_new.reshape(-1, 1), x2_new.reshape(-1, 1),
                       v2_new.reshape(-1, 1)))  # returns new states, n x 8

    return X_new


def make_payoff_n(X, n_actions):
    '''

    :param X: n x n_actions matrix
    :param n_actions: number of actions in action space
    :return: (n_actions * n_actions x n) matrix
    '''

    # make a list of ascending states based on actions
    Xs = X[:, 1:3]
    Ys = X[:, 3:]

    pairs = np.array(list(product(Xs, Ys)))

    X_return = pairs.reshape(-1, 4)
    X_return = np.hstack((X[0, 0] * np.ones((n_actions ** 2, 1)), X_return))

    return X_return


def dynamic_optimal(x, u, d, DT):
    # p = x[:, -1].reshape(-1, 1)
    # if t >= 0.5:
    #     u = np.zeros_like(p)
    # else:
    #     u = 2*p - 1
    #
    # d = u

    # U = np.array([-np.ones_like(x[..., 2]).reshape(-1, 1), np.ones_like(x[..., 2]).reshape(-1, 1)]).T.squeeze()
    # U = np.array([-np.ones_like(x[..., 2]).reshape(-1, 1), np.zeros_like(x[..., 2]).reshape(-1, 1),
    #               np.ones_like(x[..., 2]).reshape(-1, 1)]).T.squeeze()
    # d1, d2 = get_p2_a(x, d)
    # D = np.vstack((d1, np.zeros_like(d1), d2)).T   # add zero to action
    dt = DT
    x1_new = x[..., 1].reshape(-1, 1) + (x[..., 2] * dt + 0.5 * u * dt ** 2).reshape(-1, 1)
    x2_new = x[..., 3].reshape(-1, 1) + (x[..., 4] * dt + 0.5 * d * dt ** 2).reshape(-1, 1)
    v1_new = (x[..., 2] + u * dt).reshape(-1, 1)
    v2_new = (x[..., 4] + d * dt).reshape(-1, 1)

    # v_new = np.array([x[..., 2] + a for a in da_v]).T.reshape(-1, 1)
    # p_new = np.multiply(x[:, 3].reshape(-1, 1),
    #                     np.ones((x.shape[0], 4))).reshape(-1, 1)
    X_new = np.hstack((x1_new.reshape(-1, 1), v1_new.reshape(-1, 1), x2_new.reshape(-1, 1),
                       v2_new.reshape(-1, 1)))  # returns new states, n x 8

    return X_new
