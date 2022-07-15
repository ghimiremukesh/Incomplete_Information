import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modules, diff_operators
import time
import torch
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def value_action(X_nn, t_nn, model, theta):
    d1 = X_nn[0]
    v1 = X_nn [1]
    d2 = X_nn[2]
    v2 = X_nn[3]
    p = X_nn[4]

    X = np.vstack((d1, v1, d2, v2, p))
    X = torch.tensor(X, dtype=torch.float32, requires_grad=True).T
    t = torch.tensor(t_nn, dtype=torch.float32, requires_grad=True)
    coords = torch.cat((t, X), dim=1)
    coords = torch.cat((coords, theta), dim=1)

    model_in = {'coords': coords.to(device)}
    model_output = model(model_in)

    x = model_output['model_in']
    y = model_output['model_out']

    jac, _ = diff_operators.jacobian(y, x)

    # partial gradient of V w.r.t. time and state
    dvdt = jac[..., 0, 0]
    dvdx = jac[..., 0, 1:]

    # unnormalize the costate for agent 1
    lam_1 = dvdx[:, :,  :1].detach()
    lam_2 = dvdx[:, :, 1:2].detach()
    lam_4 = dvdx[:, :, 2:3].detach()
    lam_5 = dvdx[:, :, 3:4].detach()
    lam_6 = dvdx[:, :, 4:5].detach()

    u_c = torch.tensor([-0.3, 0.3])
    d_c = torch.tensor([-0.1, 0.1])
    H = torch.zeros(2, 2)

    for i in range(len(u_c)):
        for j in range(len(d_c)):
            H[i, j] = lam_1.squeeze() * v1.squeeze() + lam_2.squeeze() * u_c[i].squeeze() + \
                         lam_4.squeeze() * v2.squeeze() + lam_5.squeeze() * d_c[j].squeeze() + \
                         lam_6.squeeze() * torch.sign(u_c[i].squeeze()) - theta * u_c[i]

    u_index = torch.argmin(H[:, :], dim=1)[0] # maximin
    d_index = torch.argmax(H[u_index, :])
    u = u_c[u_index]
    d = d_c[d_index]

    return u, d, y


def dynamic(X_nn, dt, action):
    u1, u2 = action
    v1 = X_nn[1, :] + u1 * dt
    v2 = X_nn[3, :] + u2 * dt
    d1 = X_nn[0, :] + v1 * dt
    d2 = X_nn[2, :] + v2 * dt
    p = np.clip(X_nn[4, :] + np.sign(u1) * dt, 0, 1)

    return d1, v1, d2, v2, p


if __name__ == '__main__':
    logging_root = './logs'
    ckpt_path = '../experiment_scripts/logs/soccer_hji/checkpoints/model_final.pth'
    # ckpt_path = '../logs/soccer_hji_fix/checkpoints/model_final.pth'
    activation = 'tanh'

    # Initialize and load the model
    model = modules.SingleBVPNet(in_features=7, out_features=1, type=activation, mode='mlp',
                                 final_layer_factor=1., hidden_features=32, num_hidden_layers=3)
    model.to(device)
    checkpoint = torch.load(ckpt_path)
    try:
        model_weights = checkpoint['model']
    except:
        model_weights = checkpoint
    model.load_state_dict(model_weights)
    model.eval()

    num_physical = 4
    x0 = torch.zeros(1, num_physical).uniform_(-1, 1)
    x0[:, 0] = 0.5 # put them in the center
    x0[:, 2] = 0
    x0[:, 1] = 0
    x0[:, 3] = 0

    theta = torch.Tensor([[1]])  # type L or R
    p = torch.zeros(1, 1).uniform_(0, 1)
    p = torch.Tensor([[1]]) # force prior for debugging
    X0 = torch.cat((x0, p), dim=1)
    X0 = torch.cat((X0, theta), dim=1)

    N = 151*1
    Time = np.linspace(0, 1, num=N)
    dt = Time[1] - Time[0]
    Time = np.flip(Time)

    d1 = np.zeros((N,))
    v1 = np.zeros((N,))
    u1 = np.zeros((N,))
    d2 = np.zeros((N,))
    v2 = np.zeros((N,))
    u2 = np.zeros((N,))
    p = np.zeros((N,))
    V = np.zeros((N,))

    d1[0] = X0[:, 0]
    v1[0] = X0[:, 1]
    d2[0] = X0[:, 2]
    v2[0] = X0[:, 3]
    p[0] = X0[:, 4]

    start_time = time.time()

    for j in range(1, Time.shape[0] + 1):
        X_nn = np.array([[d1[j - 1]], [v1[j - 1]], [d2[j - 1]], [v2[j - 1]], [p[j-1]]])
        t_nn = np.array([[Time[j - 1]]])
        u1[j - 1], u2[j - 1], V[j - 1] = value_action(X_nn, t_nn, model, theta)
        if j == Time.shape[0]:
            break
        else:
            d1[j], v1[j], d2[j], v2[j], p[j] = dynamic(X_nn, dt, (u1[j - 1], u2[j - 1]))
            last_action = (u1[j - 1], u2[j - 1])

    print()
    time_spend = time.time() - start_time
    print('Total solution time: %1.1f' % (time_spend), 'sec')
    print()
    Time = np.flip(Time)
    fig, ax = plt.subplots(nrows=5, ncols=1)
    ax[0].plot(Time, d1)
    ax[0].set_ylabel('Attacker')
    ax[1].plot(Time, d2)
    ax[1].set_ylabel('Defender')
    ax[2].plot(Time, p)
    ax[2].set_ylabel('Belief')
    # plt.plot(p)
    ax[3].plot(Time, u1)
    ax[3].set_ylabel('$u_A$')
    ax[4].plot(Time, u2)
    ax[4].set_ylabel('$u_D$')
    ax[4].set_xlabel('Time')

    fig2, ax2= plt.subplots(1, 1)
    ax2.plot(d1,d2)

    fig3, ax3 = plt.subplots(1, 1)
    ax3.plot(Time, V, label="NN Value")
    ax3.set_ylabel('Value')
    ax3.set_xlabel('TIme')

    val = -(d1 - d2) - (theta.detach().cpu().numpy() * u1).reshape(-1,) # plot value
    print(val)
    ax3.plot(Time, val, label="true value")
    ax3.legend()
    plt.show()