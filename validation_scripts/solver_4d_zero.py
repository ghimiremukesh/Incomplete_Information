# solve the optimization problem
import numpy as np
import multiprocessing as mp
import time
import concurrent.futures
from itertools import product, zip_longest, repeat

import torch
import utils

from utils import dynamics, U, D, DT, final_value_minmax


def optimization(p, v_curr, curr_x, t, model, DT):
    def constraint(var):
        lam_1 = var[0]
        lam_2 = 1 - lam_1
        p_1 = var[1]
        check = True
        if lam_1 == 1:
            p_2 = 1 - p_1
            check = p_1 == p
        else:
            p_2 = (p - lam_1 * p_1) / lam_2

        return (0 <= p_2 <= 1) and check

    def objective(var):
        # here, V_curr is the value at the current state
        # V_next is the max min value at the previous state leading to current state
        lam_1 = var[0]
        lam_2 = 1 - lam_1
        p_1 = var[1]
        if lam_1 == 1:
            p_2 = 1 - p_1
        else:
            p_2 = (p - lam_1 * p_1) / (lam_2)


        lam_j = np.array([[lam_1], [lam_2]])
        v_next = np.zeros((2, 1))

        p_next_1 = p_1 * torch.ones((9, 1))
        p_next_2 = p_2 * torch.ones((9, 1))
        X_new = torch.from_numpy(utils.dynamics_4d_follow_zero(curr_x, 1, 1, DT=DT))
        X_new = torch.cat((t * torch.ones(X_new.shape[0], 1), X_new), dim=1)
        X_new = np.array([utils.make_payoff_zero(X_new[0, :], X_new[1, :], X_new[2, :])]).reshape(-1, X_new.shape[-1])
        X_new = torch.from_numpy(X_new.astype(np.float32))
        x_next_1 = torch.hstack((X_new, p_next_1))
        x_next_2 = torch.hstack((X_new, p_next_2))



        # x_next_1 = torch.from_numpy(dynamics(curr_x, U, D, DT=DT))
        # p_next_1 = p_1 * torch.ones((4, 1))
        # p_next_2 = p_2 * torch.ones((4, 1))
        # t_next = t * torch.ones((4, 1))
        # x_next_2 = torch.cat((t_next, x_next_1, p_next_2), dim=1)
        # x_next_1 = torch.cat((t_next, x_next_1, p_next_1), dim=1)

        # utils.final_value_minmax((X[:, 1] - X[:, 3]), X[:, -1]).numpy()

        if t == 0:
            v_next_1 = final_value_minmax(x_next_1.numpy()[:, 1] - x_next_1.numpy()[:, 3],
                                          p_next_1.numpy()[0]).reshape(-1, 3, 3)
            v_next_2 = final_value_minmax(x_next_2.numpy()[:, 1] - x_next_2.numpy()[:, 3],
                                          p_next_2.numpy()[0]).reshape(-1, 3, 3)

        else:
            x_next_1 = {'coords': x_next_1}
            v_next_1 = model(x_next_1)['model_out'].detach().cpu().numpy().reshape(-1, 3, 3)

            x_next_2 = {'coords': x_next_2}
            v_next_2 = model(x_next_2)['model_out'].detach().cpu().numpy().reshape(-1, 3, 3)

        # do maximin on v_next
        v_next[0] = np.min(np.max(v_next_1, 2), 1)
        v_next[1] = np.min(np.max(v_next_2, 2), 1)

        # return lam_1, p_1, abs((v_curr - np.matmul(lam_j.T, v_next)).item())  # \sum_j \lambda_j v(t=k+1, x', p_j)
        return abs((v_curr['model_out'].detach().cpu().numpy() - np.matmul(lam_j.T, v_next)).item())

    lam = np.linspace(0, 1, 100)
    grid = product(lam, repeat=2)
    reduced = filter(constraint, grid)
    res = min(reduced, key=objective)
    # l_1 = float('inf')
    # p_1 = float('inf')
    # curr_min = float('inf')
    # # ini = time.time()
    # mp.set_start_method('spawn')
    # with mp.Pool(mp.cpu_count()) as pool:
    #     res = pool.imap_unordered(objective, reduced)
    #
    #     for lam_1, P_1, val in res:
    #         if val < curr_min:
    #             curr_min = val
    #             l_1 = lam_1
    #             p_1 = P_1
    # # out = time.time()
    # # print(out-ini)
    # res = (l_1, p_1)

    # # chat gpt's solution
    # # Generate a grid of values for lambda1 and lambda2
    # lam1, lam2 = np.meshgrid(np.linspace(1e-6, 0.999999, 200), np.linspace(1e-6, 0.999999, 200))
    #
    # # Flatten the grid of values into a single array
    # grid = np.stack((lam1, lam2), axis=-1).reshape(-1, 2)
    #
    # # Filter the grid to only include values that satisfy the constraint
    # reduced = filter(constraint, grid)
    #
    # # Initialize variables to store the minimum value and corresponding lambda1 and lambda2 values
    # l_1 = float('inf')
    # p_1 = float('inf')
    # curr_min = float('inf')
    #
    # # Use the multiprocessing.Pool to process the objective function in parallel
    # with mp.Pool(mp.cpu_count()) as pool:
    #     res = pool.imap_unordered(objective, reduced)
    #
    #     for lam_1, P_1, val in res:
    #         if val < curr_min:
    #             curr_min = val
    #             l_1 = lam_1
    #             p_1 = P_1
    #
    # res = (l_1, p_1)
    l_1, p_1 = res
    if l_1 == 1:
        p_2 = 1 - p_1
    else:
        p_2 = (p - l_1 * p_1) / (1 - l_1)

    p_next_1 = p_1 * torch.ones((9, 1))
    p_next_2 = p_2 * torch.ones((9, 1))
    X_new = torch.from_numpy(utils.dynamics_4d_follow_zero(curr_x, 1, 1, DT=DT))
    X_new = torch.cat((t * torch.ones(X_new.shape[0], 1), X_new), dim=1)
    X_new = np.array([utils.make_payoff_zero(X_new[0, :], X_new[1, :], X_new[2, :])]).reshape(-1, X_new.shape[-1])
    X_new = torch.from_numpy(X_new.astype(np.float32))
    x_next_1 = torch.hstack((X_new, p_next_1))
    x_next_1 = x_next_1.to(torch.float32)
    x_next_2 = torch.hstack((X_new, p_next_2))
    x_next_2 = x_next_2.to(torch.float32)


    if t == 0:
        v_next_1 = final_value_minmax(x_next_1.numpy()[:, 1] - x_next_1.numpy()[:, 3], p_1).reshape(-1, 3, 3)
        v_next_2 = final_value_minmax(x_next_2.numpy()[:, 1] - x_next_2.numpy()[:, 3], p_2).reshape(-1, 3, 3)

    else:
        x_next_1 = {'coords': x_next_1}
        v_next_1 = model(x_next_1)['model_out'].detach().cpu().numpy().reshape(-1, 3, 3)

        x_next_2 = {'coords': x_next_2}
        v_next_2 = model(x_next_2)['model_out'].detach().cpu().numpy().reshape(-1, 3, 3)


    u_1 = np.argmin(np.max(v_next_1, 2), 1)[0]
    u_2 = np.argmin(np.max(v_next_2, 2), 1)[0]

    return res, u_1, u_2
