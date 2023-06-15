import copy

import torch
import diff_operators
import numpy as np
from itertools import product
from scipy.spatial import ConvexHull

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def initialize_one_sided_game(dataset):
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

    def cav_vex(values, p, type='vex'):
        lower = True if type == 'vex' else False
        ps = np.linspace(0, 1, 11)
        values = np.vstack(values).T
        cvx_vals = np.zeros((values.shape[0], 1))
        for i in range(values.shape[0]):
            value = values[i]
            points = zip(ps, value)
            hull_points = convex_hull(points, vex=lower)
            hull_points = sorted(hull_points)
            x, y = zip(*hull_points)
            num_facets = len(hull_points) - 1
            if p[i] != 1:
                s_idx = [True if x[j] <= p[i] < x[j + 1] else False for j in range(num_facets)]
            else:
                s_idx = [True if x[j] < p[i] <= x[j + 1] else False for j in range(num_facets)]
            facets = np.array(list(zip(x, x[1:])))
            val_zips = np.array(list(zip(y, y[1:])))
            P = facets[s_idx].flatten()
            vals = val_zips[s_idx].flatten()
            x1, x2 = P
            y1, y2 = vals
            # calculate the value from the equation:
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            cvx_vals[i] = slope * p[i] + intercept
        # p_idx = [list(ps).index(each) for each in p]

        return cvx_vals


    def dynamics(x, u, d):
        u = [-u, u]
        d = [-d, d]
        dt = dataset.dt
        da = np.array([a - b for a, b in product(u, d)])
        da_v = da * dt
        da_x = da * 0.5 * dt ** 2
        x_new = np.array([x[..., 1] + dt * x[..., 2] + a for a in da_x]).T.reshape(-1, 1)
        v_new = np.array([x[..., 2] + a for a in da_v]).T.reshape(-1, 1)
        # p_new = np.multiply(x[:, 3].reshape(-1, 1),
        #                     np.ones((x.shape[0], 4))).reshape(-1, 1)
        X_new = np.hstack((x_new, v_new))

        return X_new

    def one_sided(model_output, gt,
                  model_input=None):  # model input is the function that computes Value at next timestep
        source_boundary_values = gt['source_boundary_values']
        x = model_output['model_in']
        y = model_output['model_out']  # (meta_batch_size, num_points, 1)
        dirichlet_mask = gt['dirichlet_mask']
        time = dataset.time

        if time != 0.0:
            assert model_input is not None, "Need a value function for the next time-step!"
            model = model_input
            x_prev = torch.clone(x).detach().cpu().numpy()
            t_next = time - dataset.dt
            x_next = torch.from_numpy(dynamics(x_prev, dataset.umax, dataset.dmax))
            x_next = torch.cat((t_next * torch.ones((x_next.shape[0], 1)), x_next), dim=1)
            # coords_in = {'coords_in': torch.tensor(x_next).to(device)}
            vs = []
            ps = np.linspace(0, 1, dataset.num_ps)
            p = x_prev[..., -1]
            for p_each in ps:
                p_next = p_each * torch.ones_like(x_next[:, 1]).reshape(-1, 1)
                x_next_p = torch.cat((x_next, p_next), dim=1)
                coords_in = {'coords': x_next_p.to(device)}
                v_next = model(coords_in)['model_out'].detach().cpu().numpy()
                v_next = v_next.reshape(-1, 2, 2)
                v_next = np.min(np.max(v_next, 2), 1)
                vs.append(v_next)

            true_v = torch.tensor(cav_vex(vs, p.flatten(), type='vex')).reshape(1, -1, 1).to(device)  # vex for convexification
            td_error = y[~dirichlet_mask] - true_v[~dirichlet_mask]

        else:
            td_error = torch.tensor([0.])

        dirichlet = y[dirichlet_mask] - source_boundary_values[dirichlet_mask]

        return {'dirichlet': torch.abs(dirichlet).sum(),
                'td_error': torch.abs(td_error).sum()}

    return one_sided


def initialize_val_grad_loss(dataset):
    def loss(model_output, gt):
        gt_values = gt['values']
        grad = gt['gradient']

        value = model_output['model_out']
        x = model_output['model_in']
        # jac, _ = diff_operators.jacobian(value, x)
        y_flat = value.view(-1, 1)
        jac = torch.autograd.grad(y_flat, x, torch.ones_like(y_flat), create_graph=True)[0]
        dvdp = jac[..., -1].reshape(-1, 1)

        value_diff = value - gt_values
        grad_diff = grad - dvdp

        factor = (torch.abs(value_diff).sum()/torch.abs(grad_diff).sum()).detach()

        return {'value error': torch.abs(value_diff).sum(),
                'gradient error': factor * torch.abs(grad_diff).sum()}

    return loss

