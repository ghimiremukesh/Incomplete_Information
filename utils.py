# utility functions
from scipy.spatial import ConvexHull
import numpy as np
from scipy.spatial import convex_hull_plot_2d
import matplotlib.pyplot as plt
import os
from itertools import product

U = 1
D = 0.8
DT = 0.5


def dynamics(x, u, d, DT):
    u = [-u, u]
    d = [-d, d]
    dt = DT
    da = np.array([a - b for a, b in product(u, d)])
    da_v = da * dt
    da_x = da * 0.5 * dt ** 2
    x_new = np.array([x[..., 1] + dt * x[..., 2] + a for a in da_x]).T.reshape(-1, 1)
    v_new = np.array([x[..., 2] + a for a in da_v]).T.reshape(-1, 1)
    # p_new = np.multiply(x[:, 3].reshape(-1, 1),
    #                     np.ones((x.shape[0], 4))).reshape(-1, 1)
    X_new = np.hstack((x_new, v_new))

    return X_new


def dynamics_4d(x, u, d, DT):
    def get_p2_a(x, d_max):
        v = x[..., 4]  # velocity for p2
        a_l = -d_max * (1 - np.maximum(np.zeros_like(v), 0.5 * np.sign(v)))
        a_r = d_max * (1 - np.maximum(np.zeros_like(v), 0.5 * np.sign(-v)))

        return a_l, a_r

    U = np.array([-np.ones_like(x[..., 2]).reshape(-1, 1)*u, np.ones_like(x[..., 2]).reshape(-1, 1)*u]).T.squeeze()
    d1, d2 = get_p2_a(x, d)
    D = np.vstack((d1, d2)).T
    dt = DT
    x1_new = x[..., 1].reshape(-1, 1) + x[..., 2].reshape(-1, 1) * dt + 0.5 * U * dt**2
    x2_new = x[..., 3].reshape(-1, 1) + x[..., 4].reshape(-1, 1) * dt + 0.5 * D * dt**2
    v1_new = x[..., 2].reshape(-1, 1) + U * dt
    v2_new = x[..., 4].reshape(-1, 1) + D * dt

    # v_new = np.array([x[..., 2] + a for a in da_v]).T.reshape(-1, 1)
    # p_new = np.multiply(x[:, 3].reshape(-1, 1),
    #                     np.ones((x.shape[0], 4))).reshape(-1, 1)
    X_new = np.hstack((x1_new.reshape(-1, 1), v1_new.reshape(-1, 1), x2_new.reshape(-1, 1), v2_new.reshape(-1, 1)))  # returns new states, n x 8

    return X_new

def dynamics_4d_follow_zero(x, u, d, DT):
    def get_p2_a(x, d_max):
        v = x[..., 4]  # velocity

        a_l = np.zeros_like(v)
        a_r = np.zeros_like(v)

        try:
            a_l[np.where(v == 0)] = -d_max
            a_r[np.where(v == 0)] = d_max
            a_l[np.where(v < 0)] = -d_max
            a_r[np.where(v < 0)] = 0.5 * d_max

            a_l[np.where(v > 0)] = -0.5 * d_max
            a_r[np.where(v > 0)] = d_max
        except:
            a_l = -d_max if v < 0 else -0.5 * d_max
            a_r = 0.5 * d_max if v < 0 else d_max

        return a_l, a_r

    U = np.array([-np.ones_like(x[..., 2]).reshape(-1, 1), np.zeros_like(x[..., 2]).reshape(-1, 1), np.ones_like(x[..., 2]).reshape(-1, 1)]).T.squeeze()
    d1, d2 = get_p2_a(x, d)
    D = np.vstack((d1, np.zeros_like(d1), d2)).T
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

def dynamics_4d_follow(x, u, d, DT):
    def get_p2_a(x, d_max):
        v = x[..., 4]  # velocity

        a_l = np.zeros_like(v)
        a_r = np.zeros_like(v)
        try:
            a_l[np.where(v == 0)] = -d_max
            a_r[np.where(v == 0)] = d_max
            a_l[np.where(v < 0)] = -d_max
            a_r[np.where(v < 0)] = 0.5 * d_max

            a_l[np.where(v > 0)] = -0.5 * d_max
            a_r[np.where(v > 0)] = d_max
        except:
            a_l = -d_max if v < 0 else -0.5 * d_max
            a_r = 0.5 * d_max if v < 0 else d_max

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

def dynamics_zero_relative_high(x, u, d, DT):
    def get_p2_a(x, d_max):
        v = x[..., 3]  # p2's velocity

        a_l = np.zeros_like(v)
        a_r = np.zeros_like(v)


        # a_l[np.where(v < 0)] = -d_max
        # a_r[np.where(v < 0)] = 0.5 * d_max
        #
        # a_l[np.where(v > 0)] = -0.5 * d_max
        # a_r[np.where(v > 0)] = d_max

        try:
            a_l[np.where(v == 0)] = -d_max
            a_r[np.where(v == 0)] = d_max
            a_l[np.where(v < 0)] = -d_max
            a_r[np.where(v < 0)] = 0.5 * d_max
            a_l[np.where(v > 0)] = -0.5 * d_max
            a_r[np.where(v > 0)] = d_max

        except:
            a_l = -d_max if v <= 0 else -0.5 * d_max
            a_r = 0.5 * d_max if v < 0 else d_max

        return a_l, a_r

    U = np.array([-np.ones_like(x[..., 2]).reshape(-1, 1), np.zeros_like(x[..., 2]).reshape(-1, 1), np.ones_like(x[..., 2]).reshape(-1, 1)]).T.squeeze()
    d1, d2 = get_p2_a(x, d)
    D = np.vstack((d1, np.zeros_like(d1), d2)).T
    dt = DT
    da = []
    for i in range(len(U)):
        da.append([np.array([a - b for a, b in product(U[i, :], D[i, :])])])

    da = np.array(da)
    da_x = da * 0.5 * dt ** 2
    x_new = np.array([x[i, 1].reshape(-1, 1) + dt * (x[i, 2] - x[i, 3]).reshape(-1, 1) + da_x[i]
                      for i in range(len(da_x))]).reshape(-1, 1)
    v1_new = x[..., 2].reshape(-1, 1) + U * dt
    v2_new = x[..., 3].reshape(-1, 1) + D * dt


    # v_new = np.array([x[..., 2] + a for a in da_v]).T.reshape(-1, 1)
    # p_new = np.multiply(x[:, 3].reshape(-1, 1),
    #                     np.ones((x.shape[0], 4))).reshape(-1, 1)
    X_new = np.hstack((x_new.reshape(-1, 1), v1_new.repeat(3).reshape(-1, 1),
                       np.tile(v2_new, 3).reshape(-1, 1)))  # returns new states, n x 8

    return X_new

def dynamics_zero_relative(x, u, d, DT):
    def get_p2_a(x, d_max):
        v = x[..., 3]  # p2's velocity

        a_l = np.zeros_like(v)
        a_r = np.zeros_like(v)

        try:
            a_l[np.where(v == 0)] = -d_max
            a_r[np.where(v == 0)] = d_max
            a_l[np.where(v < 0)] = -d_max
            a_r[np.where(v < 0)] = 0.5 * d_max

            a_l[np.where(v > 0)] = -0.5 * d_max
            a_r[np.where(v > 0)] = d_max
        except:
            a_l = -d_max if v <= 0 else -0.5 * d_max
            a_r = 0.5 * d_max if v < 0 else d_max

        return a_l, a_r

    U = np.array([-np.ones_like(x[..., 2]).reshape(-1, 1), np.zeros_like(x[..., 2]).reshape(-1, 1),
                  np.ones_like(x[..., 2]).reshape(-1, 1)]).T.squeeze()
    d1, d2 = get_p2_a(x, d)
    D = np.vstack((d1, np.zeros_like(d1), d2)).T
    dt = DT
    da = [np.array([a - b for a, b in product(U, D)])]

    da = np.array(da)
    da_x = da * 0.5 * dt ** 2
    # x_new = np.array([x[i, 1].reshape(-1, 1) + dt * (x[i, 2] - x[i, 3]).reshape(-1, 1) + da_x[i]
    #                   for i in range(len(da_x))]).T.reshape(-1, 1)

    x_new = np.array([x[..., 1] + dt * (x[..., 2] - x[..., 3]) + a for a in da_x]).T.reshape(-1, 1)

    v1_new = x[..., 2].reshape(-1, 1) + U * dt
    v2_new = x[..., 3].reshape(-1, 1) + D * dt

    # v_new = np.array([x[..., 2] + a for a in da_v]).T.reshape(-1, 1)
    # p_new = np.multiply(x[:, 3].reshape(-1, 1),
    #                     np.ones((x.shape[0], 4))).reshape(-1, 1)
    X_new = np.hstack((x_new.reshape(-1, 1), v1_new.repeat(3).reshape(-1, 1),
                       v2_new.repeat(3).reshape(-1, 1)))  # returns new states, n x 8

    return X_new
    # def get_p2_a(x, d_max):
    #     v = x[..., 3]  # p2's velocity
    #
    #     a_l = np.zeros_like(v)
    #     a_r = np.zeros_like(v)
    #
    #     try:
    #         a_l[np.where(v == 0)] = -d_max
    #         a_r[np.where(v == 0)] = d_max
    #         a_l[np.where(v < 0)] = -d_max
    #         a_r[np.where(v < 0)] = 0.5 * d_max
    #
    #         a_l[np.where(v > 0)] = -0.5 * d_max
    #         a_r[np.where(v > 0)] = d_max
    #     except:
    #         a_l = -d_max if v < 0 else -0.5 * d_max
    #         a_r = 0.5 * d_max if v < 0 else d_max
    #
    #     return a_l, a_r
    #
    # U = np.array([-np.ones_like(x[..., 2]).reshape(-1, 1), np.zeros_like(x[..., 2]).reshape(-1, 1), np.ones_like(x[..., 2]).reshape(-1, 1)]).T.squeeze()
    # d1, d2 = get_p2_a(x, d)
    # D = np.vstack((d1, np.zeros_like(d1), d2)).T
    # dt = DT
    # da = [a - b for a, b in product(U, D)]
    #
    # da = np.array(da)
    # da_x = da * 0.5 * dt ** 2
    # x_new = x[1] + dt * (x[2] - x[3]) + da_x
    #
    # # x_new = np.array([x[..., 1] + dt * (x[..., 2] - x[..., 3]) + a for a in da_x]).T.reshape(-1, 1)
    #
    # v1_new = x[2] + U * dt
    # v2_new = x[3] + D * dt
    #
    #
    # # v_new = np.array([x[..., 2] + a for a in da_v]).T.reshape(-1, 1)
    # # p_new = np.multiply(x[:, 3].reshape(-1, 1),
    # #                     np.ones((x.shape[0], 4))).reshape(-1, 1)
    # X_new = np.hstack((x_new.reshape(-1, 1), v1_new.repeat(3).reshape(-1, 1),
    #                    np.tile(v2_new, 3).reshape(-1, 1)))  # returns new states, n x 8
    #
    # return X_new

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

def make_payoff_zero(x, y, z):
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
    # pairs = np.array([(x, y) for x in X for y in Y])

    X_return = pairs.reshape(-1, 4)
    X_return = np.hstack((x[0] * np.ones((9, 1)), X_return))

    return X_return

def go_forward(x, u, d, DT):
    a = u - d
    dt = DT
    pos = x[0]
    vel = x[1]
    pos_new = pos + vel * dt + 0.5 * a * dt ** 2
    vel_new = vel + a * dt

    return np.array([pos_new, vel_new])

def go_forward_4d(x, u, d, DT):
    x1 = x[1]
    v1 = x[2]
    x2 = x[3]
    v2 = x[4]

    x1_new = x1 + v1 * DT + 0.5 * u * DT ** 2
    v1_new = v1 + u * DT
    x2_new = x2 + v2 * DT + 0.5 * d * DT ** 2
    v2_new = v2 + d * DT

    return np.array([x1_new, v1_new, x2_new, v2_new])

def go_forward_relative(x, u, d, DT):
    x1 = x[1]
    v1 = x[2]
    v2 = x[3]

    x1_new = x1 + (v1 - v2) * DT + 0.5 * (u-d) * DT ** 2
    v1_new = v1 + u * DT
    v2_new = v2 + d * DT

    return np.array([x1_new, v1_new, v2_new])

def get_p2_a_follow(x, d_max):
    v = x[..., 4]  # velocity

    a_l = 0
    a_r = 0

    a_l = -d_max if v < 0 else -0.5 * d_max
    a_r = 0.5 * d_max if v < 0 else d_max

    return a_l, a_r

def get_p2_a_follow_relative(x, d_max):
    v = x[..., 3]

    a_l = 0
    a_r = 0

    a_l = -d_max if v <= 0 else -0.5 * d_max
    a_r = 0.5 * d_max if v < 0 else d_max

    return a_l, a_r



def get_p2_a(x, d_max):
    v = x[..., 4]  # velocity for p2
    a_l = -d_max * (1 - np.maximum(np.zeros_like(v), 0.5 * np.sign(v)))
    a_r = d_max * (1 - np.maximum(np.zeros_like(v), 0.5 * np.sign(-v)))

    return [a_l, a_r]

def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


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


def sym(p):
    if p <= 0.4:
        return 2 * p
    elif 0.4 < p <= 0.5:
        return 2 - 2 * p
    elif 0.5 < p <= 0.6:
        return 2 * p - 2
    else:
        return 0.2 - 2 * p


def f(x, p):
    if p <= 0.4:
        return np.ones_like(x) * p
    elif p <= 0.5:
        return np.ones_like(x) - p
    elif p <= 0.75:
        return np.ones_like(x) * p
    else:
        return -np.ones_like(x) * p


def g(x, p):
    if p < 0.5:
        return 0.5 * x - p
    else:
        return p - 0.5 * x


def h(x, p):
    if p < 0.4:
        return x * p
    elif 0.4 <= p <= 0.6:
        return 0.5 * x * p
    else:
        return np.ones_like(x)


def final_value(x, p):
    return p * np.maximum(0, x) + (1 - p) * np.maximum(0, -x)


def final_value_minmax(x, p):
    return 10 * (-p * np.maximum(0, x) - (1 - p) * np.maximum(0, -x))



# def cvx_hull(values, p):
#     values = np.vstack(values).T  # transpose into N x 20 matrix -- each 1 x 20 contains values between p [0, 1]
#     cvx_vals = values
#     ps = np.linspace(0, 1, ).reshape(-1, 1)
#     for i in range(values.shape[0]):  # values.shape[0]
#         points = np.hstack((ps, values[i, :].reshape(-1, 1)))
#         try:
#             hull = ConvexHull(points)
#             # convex_hull_plot_2d(hull)
#             p_min = points[hull.vertices[2:]][1, 0]
#             p_max = points[hull.vertices[2:]][0, 0]
#             if p_min <= p[i] <= p_max:
#                 cvx_vals[i, :] = points[hull.vertices[2:]][1, 1]
#         except:
#             pass
#
#     p_idx = [list(ps).index(each) for each in p]
#
#     return cvx_vals[np.arange(len(cvx_vals)), p_idx]
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


def cvx_hull(values, p):
    values = np.vstack(values).T  # transpose into N x 20 matrix -- each 1 x 20 contains values between p [0, 1]
    # cvx_vals = values
    cvx_vals = []
    max_val = 2 * np.max(values, axis=1).reshape(-1, 1)
    values = np.hstack((values, max_val))
    for i in range(values.shape[0]):  # values.shape[0]
        ps = np.linspace(0, 1, 11)
        ps = np.append(ps, ps[np.argmax(values[i, :-1])])
        points = np.hstack((ps.reshape(-1, 1), values[i, :].reshape(-1, 1)))
        try:
            hull = ConvexHull(points, qhull_options='QG' + str(11))  # take convex hull instead of the last one
            eqs = hull.equations[hull.good]
            simplices = hull.simplices[hull.good]
            # num_facets = len(simplices)
            ranges = sorted([sorted(each) for each in simplices])
            s_idx = [True if points[a[0]][0] <= p[i] < points[a[1]][0] else False for a in
                     ranges]  # only need to find for that particular facet
            P = points[simplices[s_idx]].squeeze()
            P1, P2 = P[0], P[1]
            x1, y1 = P1
            x2, y2 = P2
            # calculate the value from the equation:
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            cvx_vals.append(slope * p[i] + intercept)

            # GET THE NUMBER OF FACETS
            #
            # # try this
            # if points[hull.vertices[0]][0] == 1:
            #     if points[hull.vertices[1]][0] == 0:
            #         # then there is one straight line
            #         # find the equation of the line
            #         idx = np.argmin(hull.equations[:, 2])
            #         slope = -hull.equations[idx][0] / hull.equations[idx][1]
            #         intercept = -hull.equations[idx][2] / hull.equations[idx][1]
            #
            #         cvx_vals[i, :] = (slope * ps + intercept).reshape(-1, )
            #
            #     else:  # not a straight line, find two points and connect
            #         b = points[hull.vertices[1]][0]
            #         x1 = b[0]  # pmax
            #         y1 = b[1]
            #         a = points[hull.vertices[2]][0]
            #         x2 = a[0]  # pmin
            #         y2 = a[1]
            #
            #         slope = (y2 - y1) / (x2 - x1)
            #
            #         if x2 <= p[i] <= x1:
            #             cvx_vals[i, :] = (y1 - slope * x1 + slope * ps).reshape(-1, )

            # convex_hull_plot_2d(hull)
            # p_min = points[hull.vertices[2:]][1, 0]
            # p_max = points[hull.vertices[2:]][0, 0]
            # if p_min <= p[i] <= p_max:
            #     cvx_vals[i, :] = points[hull.vertices[2:]][1, 1]
            # find the topmost simplex index using the offset value
            # cav_i = np.argmax(hull.equations[:, 2])
            # get the equation of line
        except:
            pass

    # p_idx = [list(ps).index(each) for each in p]

    return cvx_vals  # cvx_vals[np.arange(len(cvx_vals)), p_idx]
