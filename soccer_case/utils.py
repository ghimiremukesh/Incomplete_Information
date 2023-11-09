import numpy as np
import math
from tqdm import tqdm
from itertools import product, chain
import os

# SCALING_FACTOR = 50.
# SAFETY_RADIUS = 0.05  # 0.1 meters
# for hexner's comparison
GOAL_1 = (0, 1)
GOAL_2 = (0, -1)

# for constrained case train
# GOAL_1 = (0.25, 0.4)
# GOAL_2 = (0.25, -0.4)

# GOAL_1 = (1, 0)
# GOAL_2 = (-1, 0)




def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def unicycle_dynamics(x, y, theta, u, w, dt=0.1):
    """
    Unicycle dynamical model for the players with velocity control
    :param x: x-coordinate of the player
    :param y: y-coordinate of the player
    :param theta: orientation of the player w.r.t +ve x-axis
    :param u: linear velocity control
    :param w: angular velocity control
    :param dt: time-discretization, default is 0.1
    :return: new state (x, y, theta)
    """
    xdot = u * np.cos(theta)
    ydot = u * np.sin(theta)
    thetadot = w

    x_new = x + xdot * dt
    y_new = y + ydot * dt
    theta_new = theta + thetadot * dt

    return (x_new, y_new, theta_new)


def point_dynamics(X, U, dt=0.1):
    """
    Point dynamics with velocity control in x and y direction
    :param X: State (x, y, vx, vy)
    :param U: Controls (ux, uy)
    :return: new state: (x, y, vx, vy)
    """

    x = X[0]
    y = X[1]
    vx = X[2]
    vy = X[3]
    ux = U[0]
    uy = U[1]

    xdot = vx
    ydot = vy
    vxdot = ux
    vydot = uy

    x_new = x + xdot * dt + 0.5 * ux * dt ** 2
    y_new = y + ydot * dt + 0.5 * uy * dt ** 2
    vx_new = vx + vxdot * dt
    vy_new = vy + vydot * dt

    return x_new, y_new, vx_new, vy_new

def point_dynamics_velocity(X, U, dt=0.1):
    """
    Point dynamics with velocity control

    :param X: State (x, y)
    :param U: Controls (vx, vy)
    :param dt: time discretization
    :return: new state (x, y)
    """

    x = X[0]
    y = X[1]

    vx = U[0]
    vy = U[1]

    x_new = x + vx * dt
    y_new = y + vy * dt

    return x_new, y_new

def go_forward(x, U, D, dt=0.1):

    ux, uy = U[0], U[1]
    dx, dy = D[0], D[1]

    X1 = x[..., :4]
    X2 = x[..., 4:8]

    # for p1
    x1 = X1[..., 0]
    y1 = X1[..., 1]
    vx1 = X1[..., 2]
    vy1 = X1[..., 3]


    x1dot = vx1
    y1dot = vy1
    vx1dot = ux
    vy1dot = uy

    x1_new = x1 + x1dot * dt + 0.5 * ux * dt ** 2
    y1_new = y1 + y1dot * dt + 0.5 * uy * dt ** 2
    vx1_new = vx1 + vx1dot * dt
    vy1_new = vy1 + vy1dot * dt

    # for p2
    x2 = X2[..., 0]
    y2 = X2[..., 1]
    vx2 = X2[..., 2]
    vy2 = X2[..., 3]


    x2dot = vx2
    y2dot = vy2
    vx2dot = dx
    vy2dot = dy

    x2_new = x2 + x2dot * dt + 0.5 * dx * dt ** 2
    y2_new = y2 + y2dot * dt + 0.5 * dy * dt ** 2
    vx2_new = vx2 + vx2dot * dt
    vy2_new = vy2 + vy2dot * dt

    return np.hstack((x1_new, y1_new, vx1_new, vy1_new, x2_new, y2_new, vx2_new, vy2_new))



def point_dyn_vel(x, u_max, d_max, dt=0.1):



    U = np.array([-u_max * np.ones_like(x[..., 2]).reshape(-1, 1), np.zeros_like(x[..., 2]).reshape(-1, 1),
                  u_max * np.ones_like(x[..., 2]).reshape(-1, 1)]).T.squeeze()

    D = np.array([-d_max * np.ones_like(x[..., 2]).reshape(-1, 1), np.zeros_like(x[..., 2]).reshape(-1, 1),
                  d_max * np.ones_like(x[..., 2]).reshape(-1, 1)]).T.squeeze()

    x1_new = x[..., 0].reshape(-1, 1) + U * dt
    y1_new = x[..., 1].reshape(-1, 1) + 0 * U
    x2_new = x[..., 2].reshape(-1, 1) + D * dt
    y2_new = x[..., 3].reshape(-1, 1) + 0 * D

    # v_new = np.array([x[..., 2] + a for a in da_v]).T.reshape(-1, 1)
    # p_new = np.multiply(x[:, 3].reshape(-1, 1),
    #                     np.ones((x.shape[0], 4))).reshape(-1, 1)
    X_new = np.hstack((x1_new.reshape(-1, 1), y1_new.reshape(-1, 1), x2_new.reshape(-1, 1),
                       y2_new.reshape(-1, 1)))  # returns new states, n x 8

    return X_new


def point_dyn(x, u_max, d_max, dt=0.1):
    """
    Point dynamics with acceleration control for all possible actions
    :param X: Joint state of players
    :param u_max: upper bound for control
    :param d_max: upper bound for control
    :return: new states: [X1, X2, ...., Xn] containing all possible states
    """

    # get a map of actions u_map and d_map
    us = list(product([-u_max, 0, u_max], repeat=2))
    ds = list(product([-d_max, 0, d_max], repeat=2))
    umap = {k: v for (k, v) in enumerate(us)}
    dmap = {k: v for (k, v) in enumerate(ds)}

    U = np.array([i for i in range(9)]).reshape(1, -1)
    U = np.repeat(U, x[..., 2].shape[0], axis=0)

    D = np.array([i for i in range(9)]).reshape(1, -1)
    D = np.repeat(D, x[..., 2].shape[0], axis=0)

    action_array_u = np.array([umap[i] for i in range(len(umap))])[U]
    action_array_d = np.array([dmap[i] for i in range(len(dmap))])[D]

    x1 = x[:, 0].reshape(-1, 1)
    y1 = x[:, 1].reshape(-1, 1)
    vx1 = x[:, 2].reshape(-1, 1)
    vy1 = x[:, 3].reshape(-1, 1)

    x2 = x[:, 4].reshape(-1, 1)
    y2 = x[:, 5].reshape(-1, 1)
    vx2 = x[:, 6].reshape(-1, 1)
    vy2 = x[:, 7].reshape(-1, 1)

    x1dot = vx1
    y1dot = vy1

    x2dot = vx2
    y2dot = vy2

    vx1dot = action_array_u[:, :, 0]
    vy1dot = action_array_u[:, :, 1]

    vx2dot = action_array_d[:, :, 0]
    vy2dot = action_array_d[:, :, 1]

    x1_new = x1 + x1dot * dt + 0.5 * vx1dot * (dt ** 2)
    y1_new = y1 + y1dot * dt + 0.5 * vy1dot * (dt ** 2)
    vx1_new = vx1 + vx1dot * dt
    vy1_new = vy1 + vy1dot * dt

    x2_new = x2 + x2dot * dt + 0.5 * vx2dot * (dt ** 2)
    y2_new = y2 + y2dot * dt + 0.5 * vy2dot * (dt ** 2)
    vx2_new = vx2 + vx2dot * dt
    vy2_new = vy2 + vy2dot * dt

    X_new = np.hstack((x1_new.reshape(-1, 1), y1_new.reshape(-1, 1), vx1_new.reshape(-1, 1), vy1_new.reshape(-1, 1),
                       x2_new.reshape(-1, 1), y2_new.reshape(-1, 1), vx2_new.reshape(-1, 1),
                       vy2_new.reshape(-1, 1)))

    return X_new

def check_violation(X1, X2, R):
    """
    Check for state constraint violation.

    :param X1: Player 1's state
    :param X2: Player 2's state
    :param R: Safety radius of player 1
    :return: +infty if state violation occurs, 1 otherwise
    """
    violation = np.linalg.norm(X1 - X2, axis=1) - R
    violation[violation >= 0] = 1
    violation[violation < 0] = 3  # remove state constraint to check

    return violation.reshape(-1, 1)

def inst_cost(u_max, d_max, R1, R2):
    us = list(product([-u_max, 0, u_max], repeat=2))
    ds = list(product([-d_max, 0, d_max], repeat=2))
    umap = {k: v for (k, v) in enumerate(us)}
    dmap = {k: v for (k, v) in enumerate(ds)}

    U = np.array([i for i in range(9)]).reshape(1, -1)

    D = np.array([i for i in range(9)]).reshape(1, -1)

    action_array_u = np.array([umap[i] for i in range(len(umap))])[U]
    action_array_d = np.array([dmap[i] for i in range(len(dmap))])[D]

    loss1 = np.sum(R1 * action_array_u ** 2, axis=-1)
    loss2 = np.sum(R2 * action_array_d ** 2, axis=-1)

    payoff = np.sum((list(product(loss1.flatten(), -loss2.flatten()))), axis=1)

    return payoff


def inst_cost_test(u_max, d_max):

    R1 = 1
    R2 = 10
    us = list(product([-u_max, 0, u_max], repeat=2))
    ds = list(product([-d_max, 0, d_max], repeat=2))
    umap = {k: v for (k, v) in enumerate(us)}
    dmap = {k: v for (k, v) in enumerate(ds)}

    U = np.array([i for i in range(9)]).reshape(1, -1)

    D = np.array([i for i in range(9)]).reshape(1, -1)

    action_array_u = 0.9 * np.array([umap[i] for i in range(len(umap))])[U]
    action_array_d = np.array([dmap[i] for i in range(len(dmap))])[D]

    loss1 = np.sum(R1 * action_array_u ** 2, axis=-1)
    loss2 = np.sum(R2 * action_array_d ** 2, axis=-1)

    payoff = np.sum((list(product(loss1.flatten(), -loss2.flatten()))), axis=1)

    return payoff

def final_cost(X1, X2, G, p, R=0.05, game='cons'):
    """
    Compute the payoff at the final time
    :param X1: State of player 1
    :param G: Goal positions ([g1, g2, ...., gn])
    :param p: [p_1, p_2, ..., p_n] distribution over goals
    :return: scalar cost
    """

    violation = check_violation(X1, X2, R)  # state constraint violation

    assert type(p) == np.ndarray, "p must be a numpy array"

    # we just have two goals
    g1 = np.array(G[0])
    g2 = np.array(G[1])

    dist1 = np.linalg.norm(X1 - g1, axis=1).reshape(-1, 1) ** 2
    dist2 = np.linalg.norm(X1 - g2, axis=1).reshape(-1, 1) ** 2

    # player 2
    dist1_p2 = np.linalg.norm(X2 - g1, axis=1).reshape(-1, 1) ** 2
    dist2_p2 = np.linalg.norm(X2 - g2, axis=1).reshape(-1, 1) ** 2

    # cost = np.multiply(p, dist1) + np.multiply((1 - p), dist2)

    cost = np.multiply(p, dist1) + np.multiply((1 - p), dist2) - \
           np.multiply(p, dist1_p2) - np.multiply((1 - p), dist2_p2) #+ 2  # a constant to make the value always +ve

    # cost = np.multiply(p, dist1 ** 2) + np.multiply((1 - p), dist2 ** 2) - \
    #        np.multiply(p, dist1_p2 ** 2) - np.multiply((1 - p), dist2_p2 ** 2)

    if game == 'cons':
        payoff = np.multiply(violation, cost)
    else:
        payoff = cost

    return payoff


def make_pairs(X1, X2):
    """
    Returns a matrix with all possible next states
    :param X1: states of P1
    :param X2: states of P2
    :return: X containing all pairs of (X1, X2)
    """

    # m, n = X1.shape[0], X2.shape[0]
    m, n = 9, 9
    dim = X2.shape[1]

    # Repeat and tile to create all pairs
    X1_rep = np.repeat(X1, n, axis=0)

    X2_p = X2.reshape(-1, n, dim)
    X2_rep = np.repeat(X2_p, n, axis=0).reshape(-1, dim)

    # Stack the pairs horizontally
    result = np.hstack((X1_rep, X2_rep))

    return result

def make_pairs_vel(X1, X2):
    """
    Returns a matrix with all possible next states
    :param X1: states of P1
    :param X2: states of P2
    :return: X containing all pairs of (X1, X2)
    """

    # m, n = X1.shape[0], X2.shape[0]
    m, n = 3, 3
    dim = X2.shape[1]

    # Repeat and tile to create all pairs
    X1_rep = np.repeat(X1, n, axis=0)

    X2_p = X2.reshape(-1, n, dim)
    X2_rep = np.repeat(X2_p, n, axis=0).reshape(-1, dim)

    # Stack the pairs horizontally
    result = np.hstack((X1_rep, X2_rep))

    return result


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


def cav_vex(values, type='vex', num_ps=11):
    lower = True if type == 'vex' else False
    ps = np.linspace(0, 1, num_ps)
    values = np.vstack(values).T
    cvx_vals = np.zeros((values.shape[0], num_ps))
    p = np.linspace(0, 1, num_ps)
    for i in tqdm(range(values.shape[0])):
        value = values[i]
        points = zip(ps, value)
        hull_points = convex_hull(points, vex=lower)
        hull_points = sorted(hull_points)
        x, y = zip(*hull_points)
        num_facets = len(hull_points) - 1
        for k in range(len(p)):
            if p[k] != 1:
                s_idx = [True if x[j] <= p[k] < x[j + 1] else False for j in range(num_facets)]
            else:
                s_idx = [True if x[j] < p[k] <= x[j + 1] else False for j in range(num_facets)]
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
            # cvx_vals[i] = slope * p[i] + intercept
            cvx_vals[i, k] = slope * p[k] + intercept

    return cvx_vals


def get_analytical_u(K, R, Phi, x, ztheta):
    B = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])
    u = -np.linalg.inv(R) @ B.T @ K @ x + np.linalg.inv(R) @ B.T @ K @ Phi @ (ztheta)

    return u