import torch
import diff_operators

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def initialize_soccer_hji(dataset):
    def soccer_hji(model_output, gt):

        source_boundary_values = gt['source_boundary_values']
        x = model_output['model_in']
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
        H = torch.zeros(dataset.numpoints, 2, 2)

        for i in range(len(u_c)):
            for j in range(len(d_c)):
                H[:, i, j] = lam_1.squeeze() * v1.squeeze() + lam_2.squeeze() * u_c[i].squeeze() + \
                             lam_4.squeeze() * v2.squeeze() + lam_5.squeeze() * d_c[j].squeeze() + \
                             lam_6.squeeze() * torch.sign(u_c[i].squeeze()) - theta * u_c[i]

        u = torch.zeros(dataset.numpoints)
        d = torch.zeros(dataset.numpoints)
        # pick action based on max_d min_u H
        for i in range(dataset.numpoints):
            # d_index = torch.argmax(H[i, :, :], dim=1)[1] # minimax
            # u_index = torch.argmin(H[i, :, d_index])
            # u[i] = u_c[u_index]
            # d[i] = d_c[d_index]
            u_index = torch.argmin(H[i, :, :], dim=1)[0]  # maximin
            d_index = torch.argmax(H[i, u_index, :])
            u[i] = u_c[u_index]
            d[i] = d_c[d_index]

        u = u.to(device)
        d = d.to(device)

        # calculate hamiltonian
        ham = lam_1.squeeze() * v1.squeeze() + lam_2.squeeze() * u.squeeze() + \
              lam_4.squeeze() * v2.squeeze() + lam_5.squeeze() * d.squeeze() + \
              lam_6.squeeze() * torch.sign(u.squeeze()) - theta * u.squeeze()

        # complete information
        # ham = -lam_1.squeeze() * v1.squeeze() - lam_2.squeeze() * u.squeeze() - \
        #       lam_4.squeeze() * v2.squeeze() - lam_5.squeeze() * d.squeeze()
        # dirichlet_mask is the bool array. It evaluates whether y[dirichlet_mask] is boundary condition or not
        # HJI check
        if torch.all(dirichlet_mask):
            diff_constraint_hom = torch.Tensor([0])
        else:
            # hji equation
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
