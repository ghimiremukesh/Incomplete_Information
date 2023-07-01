# utility functions
import numpy as np
import os
from itertools import product

def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def final_value(x1, v1, x2, v2, p):
    return p * ((x1 - 1) ** 2 - (x2 - 1) ** 2) + (1 - p) * ((x1 + 1) ** 2 - (x2 + 1) ** 2) + v1 ** 2 - v2 ** 2

def get_running_payoff(u, d, R1, R2, tau=0.1):
    u = np.array(u)
    d = np.array(d)
    u_d = list(product(u, d))
    payoff = [tau * (R1 * u**2 - R2 * d ** 2) for u, d in u_d]
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
    D = np.vstack((d1, np.zeros_like(d1), d2)).T   # add zero to action
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