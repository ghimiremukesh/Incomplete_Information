import torch
from torch.utils.data import Dataset
import scipy.io
import os
import math


class SoccerIncomplete(Dataset):
    def __init__(self, numpoints, velocity=0, u_max=0.5, d_max=0.3, tMin=0, tMax=1, counter_start=0,
                 counter_end=100e3, pretrain=True, pretrain_iters=2000, num_src_samples=1000, seed=0):
        super().__init__()
        torch.manual_seed(0)

        self.numpoints = numpoints
        self.velocity = velocity
        self.uMax = u_max
        self.dMax = d_max
        # self.p = torch.zeros(1).uniform_(0, 1)  # check with a bunch of p
        # self.p = p

        self.tMin = tMin
        self.tMax = tMax

        # self.num_states = 4 # d1, v1, d2, v2
        self.num_states = 2 # dx, dv
        self.N_src_samples = num_src_samples

        self.pretrain = pretrain
        self.pretrain_counter = 0
        self.counter = counter_start
        self.pretrain_iters = pretrain_iters
        self.full_count = counter_end


        # seed
        torch.manual_seed(seed)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        start_time = 0.
        # pos = torch.zeros(self.numpoints, 4).uniform_(-1, 1) # states
        pos = torch.zeros(self.numpoints, 2).uniform_(-1, 1) # dx and dv
        # self.p = torch.zeros(1).uniform_(0, 1)
        # random process p_t = p_0
        p_t = torch.zeros(self.numpoints, 1).uniform_(0, 1)

        coords = torch.cat((pos, p_t), dim=1)
        tau = 0

        if self.pretrain:
            # only sample in time around initial condition
            time = torch.ones(self.numpoints, 1) * start_time
            coords = torch.cat((time, coords), dim=1)
        else:
            # slowly grow time
            time = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax - self.tMin) *
                                                                       (self.counter / self.full_count))
            tau =  ((self.tMax - self.tMin) * (self.counter / (0.5*self.full_count)))/1e3 if self.tMax < 0.5 else 1e-3

            coords = torch.cat((time, coords), dim=1)

            # make sure we have training samples at the initial time
            coords[-self.N_src_samples:, 0] = start_time  # remove final T after boundary training.

        # if t==1 velocities must also be 0, inverted time, t=1 is initial time
        for i in range(len(coords)):
            if coords[i, 0] == 1:
                coords[i, 1] = 0  # dx = 0
                coords[i, 2] = 0  # dv = 0
                # coords[i, 4] = 0

        # boundary values for the zero sum game V(T, ., .) = \sum p_i g_i(x); g_1(x) = -g_2(x) = (d_1 - d_2)
        # boundary_values = (p * (coords[:, 1] - coords[:, 3]) + (torch.ones_like(p) - p) * (coords[:, 3] - coords[:, 1])).reshape(-1,1)
        # boundary_values = (p * (coords[:, 1] - coords[:, 3]) + (1 - p) * (
        #             coords[:, 3] - coords[:, 1])).reshape(-1, 1)
        # boundary_values = (torch.mul(p, (coords[:, 1] - coords[:, 3]).reshape(-1,1)) + torch.mul((torch.ones_like(p) - p), (coords[:, 3] - coords[:, 1]).reshape(-1,1))).reshape(-1,1)

        # for relative coordinates V(T, ., .) = p(del_x) - (1-p) (del_x)  # try with
        boundary_values = -1 * ((coords[:, -1] * coords[:, 1]) - (1 - coords[:, -1]) * coords[:, 1]).reshape(-1, 1)

        if self.pretrain:
            dirichlet_mask = torch.ones(coords.shape[0], 1) > 0
        else:
            # only enforce initial conditions around start_time
            dirichlet_mask = (coords[:, 0, None] == start_time)

        if self.pretrain:
            self.pretrain_counter += 1
        elif self.counter < self.full_count:
            self.counter += 1

        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False

        return {'coords': coords}, {'source_boundary_values': boundary_values,
                                    'dirichlet_mask': dirichlet_mask,
                                    'tau': tau}


class SoccerHJI(Dataset):
    def __init__(self, numpoints, theta, velocity=0, u_max=0.5, d_max=0.3, tMin=0, tMax=1, counter_start=0,
                 counter_end=100e3, pretrain=True, pretrain_iters=2000, num_src_samples=1000, seed=0):
        super().__init__()
        torch.manual_seed(0)

        self.numpoints = numpoints
        self.velocity = velocity
        self.uMax = u_max
        self.dMax = d_max
        self.theta = theta

        self.tMin = tMin
        self.tMax = tMax

        self.num_states = 5 # d1 v1 d2 v2

        self.N_src_samples = num_src_samples

        self.pretrain = pretrain
        self.pretrain_counter = 0
        self.counter = counter_start
        self.pretrain_iters = pretrain_iters
        self.full_count = counter_end

        # Set the seed
        torch.manual_seed(seed)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        start_time = 0.

        # uniformly sample between [-1, 1] for position [0, 1] for belief
        pos_vel = torch.zeros(self.numpoints, 4).uniform_(-1, 1)
        # pos_vel[:, 1] = 0
        # pos_vel[:, 3] = 0
        p = torch.zeros(self.numpoints, 1).uniform_(0, 1)

        coords = torch.cat((pos_vel, p), dim=1)
        type = torch.zeros(self.numpoints, 1)
        coords= torch.cat((coords, type), dim = 1)
        coords[:self.numpoints//2, -1] = 1 # half type R
        coords[self.numpoints//2:, -1] = -1 # half type L
        if self.pretrain:
            # only sample in time around initial condition
            time = torch.ones(self.numpoints, 1)*start_time
            coords = torch.cat((time, coords), dim=1)
        else:
            # slowly grow time
            time = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax-self.tMin)*
                                                                       (self.counter / self.full_count))
            coords = torch.cat((time, coords), dim=1)

            # make sure we have training samples at the initial time
            coords[-self.N_src_samples:, 0] = start_time

        # if t==1 positions and velocities must also be 0, inverted time, t=1 is initial time
        for i in range(len(coords)):
            if coords[i, 0] == 1:
                coords[i, 1] = 0
                coords[i, 2] = 0
                coords[i, 3] = 0
                coords[i, 4] = 0

        # boundary values -- zero-sum game
        # boundary_values = (-coords[:, -1] * (coords[:, 1] - coords[:, 3])).reshape(-1, 1)

        # boundary values -- general-sum game
        boundary_values_a = (coords[:, -1] * (coords[:, 1] - coords[:, 3])).reshape(-1, 1)  # attacker's boundary value
        boundary_values_b = -torch.abs(coords[:, 1] - coords[:, 3]).reshape(-1, 1)  # defender's boundary value

        boundary_values = boundary_values_a + boundary_values_b

        if self.pretrain:
            dirichlet_mask = torch.ones(coords.shape[0], 1) > 0
        else:
            # only enforce initial conditions around start_time
            dirichlet_mask = (coords[:, 0, None] == start_time)

        if self.pretrain:
            self.pretrain_counter += 1
        elif self.counter < self.full_count:
            self.counter += 1


        return {'coords': coords}, {'source_boundary_values': boundary_values,
                                    'dirichlet_mask': dirichlet_mask}


