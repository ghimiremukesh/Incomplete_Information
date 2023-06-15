# simulate the game
import numpy as np
import random

import utils
from solver_4d import optimization
from utils import final_value_minmax, go_forward
import icnn_pytorch_adaptive as icnn_pytorch
import torch

# set nature distribution
p = 0.5  # probability of being type Right
# set Player 1's type: 0 for right, 1 for left
# pick p1_type at random from the distribution p
p1_type = random.choices([0, 1], [p, 1 - p])[0]
p1_type = 1
type_map = {0: 'Right', 1: 'Left'}

DT = 0.1
# initialize two models for first two timesteps, last value is known from the function

models = [icnn_pytorch.SingleBVPNet(in_features=6, out_features=1, type='relu', mode='mlp', hidden_features=64,
                                    num_hidden_layers=3, dropout=0) for i in range(10)]

# checkpoints = [f'../experiment_scripts/logs/train_informative_11_timesteps/t_{i}/checkpoints_dir/model_final.pth'
               # for i in range(1, 11)]

checkpoints = [f'../logs/train_new_case_follow/t_{i}/checkpoints_dir/model_final.pth'
               for i in range(1, 11)]

loaded_check = [torch.load(checkpoint, map_location=torch.device("cpu")) for checkpoint in checkpoints]

try:
    model_weights = [loaded['model'] for loaded in loaded_check]
except:
    model_weights = [loaded for loaded in loaded_check]

for i in range(len(models)):
    models[i].load_state_dict(model_weights[i])
    models[i].eval()



if __name__ == "__main__":
    print(f'Player 1s Type is: {type_map[p1_type]}')
    curr_x1 = np.array([0.2, 0])  # pos and vel for player 1
    curr_x2 = np.array([-0.1, 0])
    print(f'Current position is : {curr_x1}, {curr_x2} and the belief is {p}\n')
    p_t = p
    a_map = {'0': -1, '1': 1}
    pay_size = (4, 2)

    # first get strategy for initial state (2, 2)
    t = 1  # backward time
    ts = np.arange(0, 1, DT)
 # pos and vel for player 2
    # ini_pos = torch.tensor([[t, 0, 0, p_t]], dtype=torch.float32)
    curr_pos = torch.hstack((torch.tensor(t), torch.from_numpy(curr_x1), torch.from_numpy(curr_x2), torch.tensor(p_t)))
    # curr_pos = torch.tensor(curr_pos, dtype=torch.float32)
    curr_pos = curr_pos.to(torch.float32)
    # curr_pos = ini_pos

    while t >= 0:
        model_idx = int(10 * t) - 1
        curr_model = models[model_idx]
        next_model = models[model_idx - 1]
        feed_to_model = {'coords': curr_pos}
        v_curr = curr_model(feed_to_model)
        (lam_j, p_1), u_1, u_2 = optimization(p_t, v_curr, curr_pos.numpy(), t - DT, next_model, DT=DT)

        if lam_j == 1:
            p_2 = 1 - p_1
        else:
            p_2 = (p_t - lam_j * p_1) / (1 - lam_j)
        print(f'lamda_1 = {lam_j:.2f}, p_1 = {p_1:.2f},  p_2 = {p_2:.2f}\n')

        # action selection for first splitting point (lam_1)
        # calculate probability of each action
        if p1_type == 1:
            p_i = 1 - p_t
            p_1j = 1 - p_1
            p_2j = 1 - p_2
        else:
            p_i = p_t
            p_1j = p_1
            p_2j = p_2

        if lam_j == 1:
            a0_p = 1
            a1_p = 0
        else:
            a0_p = (lam_j * p_1j) / p_i
            a1_p = ((1 - lam_j) * p_2j) / p_i

        print(f'At t = {1-t:.2f}, P1 with type {type_map[p1_type]} has the following options: \n')
        print(f'P1 could take action {a_map[str(u_1)]} with probability {a0_p:.2f} and move belief to {p_1:.2f}')
        print(f'P1 could take action {a_map[str(u_2)]} with probability {a1_p:.2f} and move belief to {p_2:.2f}\n')

        dist = [a0_p, a1_p]
        a_idx = [0, 1]
        action_idx = random.choices(a_idx, dist)[0]
        # set to calculate strategy
        # action_idx = 0
        if action_idx == 0:
            action_1 = u_1
            p_t = p_1
        else:
            action_1 = u_2
            p_t = p_2

        # for simulation purpose select p2's action randomly
        d_max = 1
        p2_ac = utils.get_p2_a_follow(curr_pos, d_max=d_max)
        p2_a = random.choices(p2_ac, [0.5, 0.5])[0]
        # to get the strategy pick one at a time
        # p2_a = -1  # -1 is l
        # p2_amap = {-1: -0.8, 1: 0.8}
        p2_action = p2_a

        print(f'P1 chooses action: {a_map[str(action_1)]} and moves the belief to p_t = {p_t:.2f}')
        print(f'P2 chooses action: {p2_action} at random\n')

        curr_x = utils.go_forward_4d(curr_pos.numpy(), a_map[str(action_1)], p2_action, DT)

        print(f'The current state is: {curr_x}\n')

        t = t - DT
        curr_pos = torch.hstack((torch.tensor(t), torch.from_numpy(curr_x), torch.tensor(p_t))).to(torch.float32)
        # curr_pos = torch.tensor([[t, curr_x[0], curr_x[1], p_t]], dtype=torch.float32)

    # final time
    print()



    # v_curr = model_0(feed_to_model)
    # (lam_j, p_1), u_1, u_2 = optimization(p_t, v_curr, ini_pos.numpy(), t-DT, model_05)
    #
    # # for debugging
    # # lam_j = 0.8
    # # p_1 = 0.33
    # if lam_j == 1:
    #     p_2 = 1 - p_1
    # else:
    #     p_2 = (p_t - lam_j * p_1) / (1 - lam_j)
    # print(f'lamda_1 = {lam_j:.2f}, p_1 = {p_1:.2f},  p_2 = {p_2:.2f}\n')
    #
    # # action selection for first splitting point (lam_1)
    # # calculate probability of each action
    # if p1_type == 1:
    #     p_i = 1 - p_t
    #     p_1j = 1 - p_1
    #     p_2j = 1 - p_2
    # else:
    #     p_i = p_t
    #     p_1j = p_1
    #     p_2j = p_2
    #
    # if lam_j == 1:
    #     a0_p = 1
    #     a1_p = 0
    # else:
    #     a0_p = (lam_j * p_1j) / p_i
    #     a1_p = ((1 - lam_j) * p_2j) / p_i
    #
    # print(f'At initial time, P1 with type {type_map[p1_type]} has the following options: \n')
    # print(f'P1 could take action {a_map[str(u_1)]} with probability {a0_p:.2f} and move belief to {p_1:.2f}')
    # print(f'P1 could take action {a_map[str(u_2)]} with probability {a1_p:.2f} and move belief to {p_2:.2f}\n')
    #
    # dist = [a0_p, a1_p]
    # a_idx = [0, 1]
    # action_idx = random.choices(a_idx, dist)[0]
    # # set to calculate strategy
    # # action_idx = 0
    # if action_idx == 0:
    #     action_1 = u_1
    #     p_t = p_1
    # else:
    #     action_1 = u_2
    #     p_t = p_2
    #
    # # for simulation purpose select p2's action randomly
    # p2_a = random.choices([-1, 1], [0.5, 0.5])[0]
    # # to get the strategy pick one at a time
    # # p2_a = -1  # -1 is l
    # p2_amap = {-1: -0.8, 1: 0.8}
    # p2_action = p2_amap[p2_a]
    #
    # print(f'P1 chooses action: {a_map[str(action_1)]} and moves the belief to p_t = {p_t:.2f}')
    # print(f'P2 chooses action: {p2_action} at random\n')
    #
    # ##########################################################################################
    # #################################### NEXT TIME-STEP ######################################
    # ##########################################################################################
    # # current time is t = 0.5
    # # next time is t = 0
    #
    # t = 0.5
    # curr_pos = go_forward(start_x, a_map[str(action_1)], p2_action, DT)
    #
    # print(f'The current relative state (pos, vel) is: {curr_pos}\n')
    #
    # pos = torch.tensor([[t, curr_pos[0], curr_pos[1], p_t]], dtype=torch.float32)
    # feed_to_model = {'coords': pos}
    # v_curr = model_05(feed_to_model)
    # (lam_j, p_1), u_1, u_2 = optimization(p_t, v_curr, pos.numpy(), t-DT, model_1)
    #
    # # for debugging
    # # lam_j = 0.8
    # # p_1 = 0.33
    # if lam_j == 1:
    #     p_2 = 1 - p_1
    # else:
    #     p_2 = (p_t - lam_j * p_1) / (1 - lam_j)
    # print(f'lamda_1 = {lam_j:.2f}, p_1 = {p_1:.2f},  p_2 = {p_2:.2f}\n')
    #
    # # action selection for first splitting point (lam_1)
    # # calculate probability of each action
    # if p1_type == 1:
    #     p_i = 1 - p_t
    #     p_1j = 1 - p_1
    #     p_2j = 1 - p_2
    # else:
    #     p_i = p_t
    #     p_1j = p_1
    #     p_2j = p_2
    #
    # if lam_j == 1:
    #     a0_p = 1
    #     a1_p = 0
    # else:
    #     a0_p = (lam_j * p_1j) / p_i
    #     a1_p = ((1 - lam_j) * p_2j) / p_i
    #
    # print(f'At Second timestep, P1 with type {type_map[p1_type]} has the following options: \n')
    # print(f'P1 could take action {a_map[str(u_1)]} with probability {a0_p:.2f} and move belief to {p_1:.2f}')
    # print(f'P1 could take action {a_map[str(u_2)]} with probability {a1_p:.2f} and move belief to {p_2:.2f}\n')
    #
    # dist = [a0_p, a1_p]
    # a_idx = [0, 1]
    # action_idx = random.choices(a_idx, dist)[0]
    # # set to calculate strategy
    # # action_idx = 0
    # if action_idx == 0:
    #     action_1 = u_1
    #     p_t = p_1
    # else:
    #     action_1 = u_2
    #     p_t = p_2
    #
    # # for simulation purpose select p2's action randomly
    # p2_a = random.choices([-1, 1], [0.5, 0.5])[0]
    # # to get the strategy pick one at a time
    # # p2_a = -1  # -1 is l
    # p2_amap = {-1: -0.8, 1: 0.8}
    # p2_action = p2_amap[p2_a]
    #
    # print(f'P1 chooses action: {a_map[str(action_1)]} and moves the belief to p_t = {p_t:.2f}')
    # print(f'P2 chooses action: {p2_action} at random\n')
    #
    # curr_pos = go_forward(curr_pos, a_map[str(action_1)], p2_action, DT)
    #
    # print(f'Final relative state (pos, vel) is: {curr_pos}.\n')
    # print(f'Payoff to the attacker is: {final_value_minmax(curr_pos[0], p_t)}')
