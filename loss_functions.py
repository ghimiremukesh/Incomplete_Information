import copy

import torch
import diff_operators
import numpy as np

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def initialize_soccer_incomplete(dataset):
    def soccer_incomplete(model_output, gt):

        source_boundary_values = gt['source_boundary_values']
        x = model_output['model_in']
        # output of the value network V(t, x, p)
        y = model_output['model_out']  # (meta_batch_size, num_points, 1)
        dirichlet_mask = gt['dirichlet_mask']

        # calculate the partial gradient w.r.t. state (not p)
        jac, _ = diff_operators.jacobian(y, x)
        dvdx = jac[..., 0, 1:-1].squeeze()  # exclude the last one
        dvdt = jac[..., 0, 0].squeeze()

        lam_d = dvdx[:, :1].detach().cpu().numpy()
        lam_v = dvdx[:, -1:].detach().cpu().numpy()

        del_v = x[:, :, 2].squeeze()


        u = np.sign(lam_v) * dataset.uMax  # bang bang control
        d = np.sign(lam_d) * dataset.dMax  # bang bang control (min H * -1)

        u = torch.as_tensor(u).squeeze().to(device)
        d = torch.as_tensor(d).squeeze().to(device)
        lam_d = torch.as_tensor(lam_d).squeeze().to(device)
        lam_v = torch.as_tensor(lam_v).squeeze().to(device)

        ham = lam_d * del_v + lam_v * (u - d)

        if torch.all(dirichlet_mask):
            diff_constraint_hom = torch.Tensor([0])
        else:
            # hji equation -dv/dt because the time is backward during training
            diff_constraint_hom = -dvdt + ham



        # boundary condition check
        dirichlet = y[dirichlet_mask] - source_boundary_values[dirichlet_mask]
        weight = 1

        weight_ratio = torch.abs(diff_constraint_hom).sum() * weight / torch.abs(dirichlet).sum()
        weight_ratio = weight_ratio.detach()
        if weight_ratio == 0:
            hjpde_weight = 1
        else:
            hjpde_weight = weight_ratio


        # A factor of (2e5, 100) to make loss roughly equal
        return {'dirichlet': torch.abs(dirichlet).sum() / weight,
                # torch.abs(dirichlet).sum(), torch.norm(torch.abs(dirichlet))
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum() / hjpde_weight}
        # return {'dirichlet': torch.abs(dirichlet).sum() * beta,  # 1e4
        #         'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}

    return soccer_incomplete


def initialize_soccer_discrete(dataset):
    def soccer_discrete(model, model_output, gt):

        source_boundary_values = gt['source_boundary_values']
        x = model_output['model_in']
        # output of the value network V(t, x, p)
        y = model_output['model_out']  # (meta_batch_size, num_points, 1)
        dirichlet_mask = gt['dirichlet_mask']

        # action candidates
        u_c = torch.tensor([-dataset.uMax, dataset.uMax])
        d_c = torch.tensor([-dataset.dMax, dataset.dMax])
        # V_next = torch.zeros(dataset.numpoints, 2, 2)
        V_next = np.zeros((dataset.numpoints, 2, 2))
        tau = gt['tau']  # time step

        x_next = [copy.deepcopy(x) for _ in range(len(u_c)) for _ in range(len(d_c))]
        # x_next = [torch.clone(x) for _ in range(len(u_c)) for j in range(len(d_c))]
        # v = [x_next[i][..., 2] + u_c[i]]

        count = 0  # brute force, try something later
        for i in range(len(u_c)):
            for j in range(len(d_c)):
                # get the next state
                v = x_next[count][..., 2] + (u_c[i] - d_c[j]) * tau
                d = x_next[count][..., 1] + v * tau
                with torch.no_grad():
                    x_next[count][..., 1] = d
                    x_next[count][..., 2] = v
                    x_next[count][..., 0] = x_next[count][..., 0] - tau
                next_in = {'coords': x_next[count]}
                V_next[:, i, j] = model(next_in)['model_out'].cpu().detach().numpy().squeeze()
                count += 1

        # array to store actual next value
        # pick action based on max_u min_d V
        # this is only for player 1 at the moment
        # u_indices = torch.argmax(torch.amin(V_next, dim=2, keepdim=True), dim=1)
        # d_indices = np.unravel_index(torch.argmin(torch.amax(V_next, dim=2, keepdim=True), dim=2), V_next.shape[0])[0][:, 0]
        u_indices = np.argmax(np.min(V_next, axis=2, keepdims=True), axis=1).flatten()
        d_indices = np.argmin(np.max(V_next, axis=1, keepdims=True), axis=2).flatten()
        # d_indices = np.unravel_index(torch.argmax(torch.amin(V_next, dim=2, keepdim=True)))
        # d_indices = torch.argmax(torch.amin(V_next, dim=2, keepdim=True), dim=2)[1]
        # d_indices = torch.argmax(V_next, dim=2)[:, 1]
        # u_indices = torch.argmin(V_next, dim=1)[:, 1]

        v_next_true = [V_next[i, u_indices[i], d_indices[i]] for i in range(len(V_next))]
        v_next_true = torch.as_tensor(v_next_true).flatten().to(device)
        # v_next_true = torch.as_tensor(np.diag(V_next[:, u_indices.flatten(),
        #                                       d_indices.flatten()].reshape(-1, 1))).to(device)
        # v_next_true = torch.diag(V_next[:, u_indices.flatten(), d_indices.flatten()]).to(device)

        # u = torch.zeros(dataset.numpoints)
        # d = torch.zeros(dataset.numpoints)
        # V_next_true = torch.zeros(dataset.numpoints)

        # remove for-loop replaced by above : keep for future debugging
        # for i in range(dataset.numpoints):
        #     d_index = torch.argmax(V_next[i, :, :], dim=1)[1]
        #     u_index = torch.argmin(V_next[i, :, d_index])
        #     u[i] = u_c[u_index]
        #     d[i] = d_c[d_index]
        #     V_next_true[i] = V_next[i, u_index, d_index]   # this is the true value of the next state from minmax

        if torch.all(dirichlet_mask):
            diff_constraint_hom = torch.Tensor([0])
        else:
            # check the value difference # for times other than terminal time
            diff_constraint_hom = y[~dirichlet_mask] - v_next_true.reshape(1, -1, 1)[~dirichlet_mask]

        # boundary condition check
        dirichlet = y[dirichlet_mask] - source_boundary_values[dirichlet_mask]

        # A factor of (2e5, 100) to make loss roughly equal
        return {'dirichlet': torch.abs(dirichlet).sum() / 150,  # 1e4
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}

    return soccer_discrete


def initialize_soccer_hji(dataset):
    def soccer_hji(model_output, gt):

        source_boundary_values = gt['source_boundary_values']
        x = model_output['model_in']

        # the network output should be V1 and V2
        y = model_output['model_out']  # (meta_batch_size, num_points, 1); value
        dirichlet_mask = gt['dirichlet_mask']
        batch_size = x.shape[1]

        # calculate the partial gradient of V w.r.t. time and state
        jac, _ = diff_operators.jacobian(y, x)
        theta = x[:, :, -1]

        # partial gradient of V w.r.t. time and state
        dvdt = jac[..., 0, 0].squeeze()
        dvdx = jac[..., 0, 1:].squeeze()

        # co-states
        lam_1 = dvdx[:, :1]
        lam_2 = dvdx[:, 1:2]
        lam_4 = dvdx[:, 2:3]
        lam_5 = dvdx[:, 3:4]
        lam_6 = dvdx[:, 4:5]

        v1 = x[:, :, 2]
        v2 = x[:, :, 4]

        # action candidates
        u_c = torch.tensor([-dataset.uMax, dataset.uMax])
        d_c = torch.tensor([-dataset.dMax, dataset.dMax])
        H1 = torch.zeros(dataset.numpoints, 2, 1)
        H2 = torch.zeros(dataset.numpoints, 2, 1)

        # for i in range(len(u_c)):
        #     for j in range(len(d_c)):
        #         H[:, i, j] = lam_1.squeeze() * v1.squeeze() + lam_2.squeeze() * u_c[i].squeeze() + \
        #                      lam_4.squeeze() * v2.squeeze() + lam_5.squeeze() * d_c[j].squeeze() + \
        #                      lam_6.squeeze() * torch.sign(u_c[i].squeeze()) - theta * u_c[i]

        # General-Sum
        for i in range(len(u_c)):
            H1[:, i] = lam_1.squeeze() * v1.squeeze() + lam_2.squeeze() * u_c[i].squeeze() + \
                       lam_6.squeeze() * u_c[i]

        for i in range(len(d_c)):
            H2[:, i] = lam_4.squeeze() * v2.squeeze() + lam_5.squeeze() * d_c[i].squeeze()

        u = torch.zeros(dataset.numpoints)
        d = torch.zeros(dataset.numpoints)
        # pick action based on max_d min_u H
        for i in range(dataset.numpoints):
            # d_index = torch.argmax(H[i, :, :], dim=1)[1] # minimax
            # u_index = torch.argmin(H[i, :, d_index])
            # u[i] = u_c[u_index]
            # d[i] = d_c[d_index]
            u_index = torch.argmax(H1[i, :, :], dim=1)  # maximin
            d_index = torch.argmin(H2[i, :, :], dim=1)
            u[i] = u_c[u_index]
            d[i] = d_c[d_index]

        u = u.to(device)
        d = d.to(device)

        # calculate hamiltonian
        # ham = lam_1.squeeze() * v1.squeeze() + lam_2.squeeze() * u.squeeze() + \
        #       lam_4.squeeze() * v2.squeeze() + lam_5.squeeze() * d.squeeze() + \
        #       lam_6.squeeze() * torch.sign(u.squeeze()) - theta * u.squeeze()

        # general-sum
        ham_1 = lam_1.squeeze() * v1.squeeze() + lam_2.squeeze() * u.squeeze() + lam_6.squeeze() * u

        ham_2 = lam_4.squeeze() * v2.squeeze() + lam_5.squeeze() * d.squeeze()

        ham = ham_1 + ham_2
        # complete information
        # ham = -lam_1.squeeze() * v1.squeeze() - lam_2.squeeze() * u.squeeze() - \
        #       lam_4.squeeze() * v2.squeeze() - lam_5.squeeze() * d.squeeze()
        # dirichlet_mask is the bool array. It evaluates whether y[dirichlet_mask] is boundary condition or not
        # HJI check
        if torch.all(dirichlet_mask):
            diff_constraint_hom = torch.Tensor([0])
        else:
            # hji equation -dv/dt because the time is backward during training
            diff_constraint_hom = -dvdt + ham
            # diff_constraint_hom = torch.max(diff_constraint_hom, (y-source_boundary_values).squeeze())
            # diff_constraint_hom = dvdt + torch.minimum(torch.tensor([[0]]), ham)
            # diff_constraint_hom = dvdt + torch.clamp(ham, max=0.0)

        # boundary condition check
        dirichlet = y[dirichlet_mask] - source_boundary_values[dirichlet_mask]

        # A factor of (2e5, 100) to make loss roughly equal
        return {'dirichlet': torch.abs(dirichlet).sum(),  # 1e4
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum() / 40}

    return soccer_hji
