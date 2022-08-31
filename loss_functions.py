import torch
import diff_operators

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
        dvdx = jac[..., 0, 1:-1].squeeze() # exclude the last one
        dvdt = jac[..., 0, 0].squeeze()

        # calculate hessian only with respect to p
        hess, _ = diff_operators.hessian(y, x)
        d2vdp = hess[..., -1, -1].squeeze()

        # co-states for hamiltonian H = argmax_u argmin_d = <\lambda, f>
        # lam_da = dvdx[:, :1].squeeze()
        # lam_va = dvdx[:, 1:2].squeeze()
        # lam_dd = dvdx[:, 2:3].squeeze()
        # lam_dv = dvdx[:, 3:4].squeeze()

        # co-states with relative coordinates
        lam_d = dvdx[:, :1].squeeze()
        lam_v = dvdx[:, -1:].squeeze()

        # v1 = x[:, :, 2].squeeze()
        # v2 = x[:, :, 4].squeeze()

        del_v = x[:, :, 2].squeeze()

        # action candidates
        u_c = torch.tensor([-dataset.uMax, dataset.uMax])
        d_c = torch.tensor([-dataset.dMax, dataset.dMax])
        H = torch.zeros(dataset.numpoints, 2, 2)

        # for i in range(len(u_c)):
        #     for j in range(len(d_c)):
        #         H[:, i, j] = lam_da * v1 + lam_va * u_c[i] + lam_dd * v2 + lam_dv * d_c[j]
        for i in range(len(u_c)):
            for j in range(len(d_c)):
                H[:, i, j] = lam_d * del_v + lam_v * (u_c[i] - d_c[j])

        u = torch.zeros(dataset.numpoints)
        d = torch.zeros(dataset.numpoints)
        # pick action based on max_u min_d H
        for i in range(dataset.numpoints):
            d_index = torch.argmax(H[i, :, :], dim=1)[1]
            u_index = torch.argmin(H[i, :, d_index])
            u[i] = u_c[u_index]
            d[i] = d_c[d_index]

        u = u.squeeze().to(device)
        d = d.squeeze().to(device)

        # ham = lam_da * v1 + lam_va * u + lam_dd * v2 + lam_dv * d
        ham = lam_d * del_v + lam_v * (u - d)

        if torch.all(dirichlet_mask):
            diff_constraint_hom = torch.Tensor([0])
        else:
            # hji equation -dv/dt because the time is backward during training
            # diff_constraint_hom = -dvdt + ham
            diff_constraint_hom = torch.min(-dvdt + ham, d2vdp)

        # boundary condition check
        dirichlet = y[dirichlet_mask] - source_boundary_values[dirichlet_mask]

        # A factor of (2e5, 100) to make loss roughly equal
        return {'dirichlet': torch.abs(dirichlet).sum()/6,  # 1e4
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}

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
        V_next = torch.zeros(dataset.numpoints, 2, 2)
        tau = 1e-3  # time step
        x_next = torch.clone(x)

        # for i in range(len(u_c)):
        #     for j in range(len(d_c)):
        #         H[:, i, j] = lam_da * v1 + lam_va * u_c[i] + lam_dd * v2 + lam_dv * d_c[j]
        for i in range(len(u_c)):
            for j in range(len(d_c)):
                # get the next state
                v = x_next[..., 2] + (u_c[i] - d_c[j]) * tau
                d = x_next[..., 1] + v * tau
                x_next[..., 1] = d
                x_next[..., 2] = v
                x_next[..., 0] = x_next[..., 0] + tau
                next_in = {'coords': x_next}
                V_next[:, i, j] = model(next_in)['model_out'].squeeze()

        u = torch.zeros(dataset.numpoints)
        d = torch.zeros(dataset.numpoints)

        # array to store actual next value
        v_next_true = torch.zeros(dataset.numpoints)
        # pick action based on max_u min_d H
        for i in range(dataset.numpoints):
            d_index = torch.argmax(V_next[i, :, :], dim=1)[1]
            u_index = torch.argmin(V_next[i, :, d_index])
            u[i] = u_c[u_index]
            d[i] = d_c[d_index]
            v_next_true[i] = V_next[i, u_index, d_index]   # this is the true value of the next state from minmax

        if torch.all(dirichlet_mask):
            diff_constraint_hom = torch.Tensor([0])
        else:
            # check the value difference
            diff_constraint_hom = y - v_next_true.reshape(-1, 1)

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


