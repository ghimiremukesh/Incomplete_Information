# solve the optimization problem
import numpy as np
import multiprocessing as mp
import time
import concurrent.futures
from itertools import product, zip_longest, repeat



import torch
import utils
# from utils import dynamics, U, D, DT, final_value_minmax


def optimization(p, v_curr, curr_x, t, model, DT, game='cons'):
    def constraint(var):
        lam_1 = var[0]
        lam_2 = 1 - lam_1
        p_1 = var[1]
        # p_2 = (p - lam_1 * p_1) / (lam_2 + 1e-9)
        p_2 = var[2]
        # check = True
        # if lam_1 == 1:
        #     p_2 = 1 - p_1
        #     check = p_1 == p
        # else:
        #     p_2 = (p - lam_1 * p_1) / lam_2
        #
        # return (0 <= p_2 <= 1) and check
        # check = 0 <= p_2 <= 1

        return ((lam_1 * p_1 + lam_2 * p_2) == p)


    def objective(var):
        # here, V_curr is the value at the current state
        # V_next is the max min value at the previous state leading to current state

        lam_1 = var[0]
        lam_2 = 1 - lam_1
        p_1 = var[1]
        p_2 = var[2]
        # if lam_1 == 1:
        #     p_2 = 1 - p_1
        # else:
        #     p_2 = (p - lam_1 * p_1) / (lam_2)

        g1 = utils.GOAL_1
        g2 = utils.GOAL_2

        G = [g1, g2]

        R1 = 0.01
        R2 = 0.01

        ts = np.around(np.linspace(0, 1, 11), 2)

        lam_j = np.array([[lam_1], [lam_2]])
        v_next = np.zeros((2, 1))

        p_next_1 = p_1 * torch.ones((81, 1))
        p_next_2 = p_2 * torch.ones((81, 1))
        # X_new = torch.from_numpy(utils.dynamics_4d_follow_zero(curr_x, 1, 1, DT=DT))
        X_new = torch.from_numpy(utils.point_dyn(np.expand_dims(curr_x, axis=0), 2, 1, dt=DT))  # input is u_max and d_max -- both are 1
        # X_new = torch.cat((t * torch.ones(X_new.shape[0], 1), X_new), dim=1)
        X_new = torch.from_numpy(utils.make_pairs(X_new[:, :4], X_new[:, 4:8]))
        x_next_1 = torch.hstack((X_new, p_next_1))
        x_next_2 = torch.hstack((X_new, p_next_2))


        if t == 0:
            v_next_1 = utils.final_cost(x_next_1[:, :2], x_next_1[:, 4:6], G, p_next_1.detach().numpy(), game=game) #+ \
                       #DT * utils.inst_cost(2, 1, R1, R2).reshape(-1, 9, 9)
            v_next_2 = utils.final_cost(x_next_2[:, :2], x_next_2[:, 4:6], G, p_next_2.detach().numpy(), game=game) #+ \
                       #DT * utils.inst_cost(2, 1, R1, R2).reshape(-1, 9, 9)
        else:
            next_1 = {'coords': x_next_1.to(torch.float32)}
            v_next_1 = model(next_1)['model_out'].detach().cpu().numpy().reshape(-1, 9, 9) #+ \
                      #DT * utils.inst_cost(2, 1, R1, R2).reshape(-1, 9, 9)

            next_2 = {'coords': x_next_2.to(torch.float32)}
            v_next_2 = model(next_2)['model_out'].detach().cpu().numpy().reshape(-1, 9, 9) #+ \
                       #DT * utils.inst_cost(2, 1, R1, R2).reshape(-1, 9, 9)


        # do maximin on v_next
        v_next[0] = np.min(np.max(v_next_1, 2), 1)
        v_next[1] = np.min(np.max(v_next_2, 2), 1)

        return abs((v_curr - np.matmul(lam_j.T, v_next)).item())

    # lam = np.linspace(0, 0.5, 1)
    lam = np.linspace(0, 1, 11)
    ps = np.linspace(0, 1, 11)
    grid = product(lam, ps, ps)
    reduced = filter(constraint, grid)
    res = min(reduced, key=objective)

    l_1, p_1, p_2 = res

    g1 = utils.GOAL_1
    g2 = utils.GOAL_2

    G = [g1, g2]

    R1 = 0.01
    R2 = 0.01


    ts = np.around(np.linspace(0, 1, 11), 2)
    # t_step = int(np.where(ts == np.round(t, 2))[0])

    p_next_1 = p_1 * torch.ones((81, 1))
    p_next_2 = p_2 * torch.ones((81, 1))
    X_new = torch.from_numpy(utils.point_dyn(np.expand_dims(curr_x, axis=0), 2, 1, dt=DT))  # input is u_max and d_max -- both are 1
    # X_new = torch.cat((t * torch.ones(X_new.shape[0], 1), X_new), dim=1)
    X_new = torch.from_numpy(utils.make_pairs(X_new[:, :4], X_new[:, 4:8]))
    x_next_1 = torch.hstack((X_new, p_next_1))
    x_next_2 = torch.hstack((X_new, p_next_2))

    if t == 0:
        v_next_1 = utils.final_cost(x_next_1[:, :2], x_next_1[:, 4:6], G, p_next_1.detach().numpy(), game=game) #+ \
                      # DT * utils.inst_cost(2, 1, R1, R2).reshape(-1, 9, 9)
        v_next_2 = utils.final_cost(x_next_2[:, :2], x_next_2[:, 4:6], G, p_next_2.detach().numpy(), game=game)#+ \
                      # DT * utils.inst_cost(2, 1, R1, R2).reshape(-1, 9, 9)
    else:
        next_1 = {'coords': x_next_1.to(torch.float32)}
        v_next_1 = model(next_1)['model_out'].detach().cpu().numpy().reshape(-1, 9, 9) #+ \
                       #DT * utils.inst_cost(2, 1, R1, R2).reshape(-1, 9, 9)

        next_2 = {'coords': x_next_2.to(torch.float32)}
        v_next_2 = model(next_2)['model_out'].detach().cpu().numpy().reshape(-1, 9, 9) #+ \
                       #DT * utils.inst_cost(2, 1, R1, R2).reshape(-1, 9, 9)

    # do maximin on v_next
    u_1 = np.argmin(np.max(v_next_1, 2))
    u_2 = np.argmin(np.max(v_next_2, 2))

    d_1 = np.argmax(v_next_1, 2).reshape(-1, )[u_1]
    d_2 = np.argmax(v_next_2, 2).reshape(-1, )[u_2]

    return res, u_1, u_2, d_1, d_2
