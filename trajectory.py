import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy.io

# from examples.choose_problem import system

# ____________________________________________________________________________________________________

bvpAA = True
bvpNANA = False
bvpANA = False
bvpNAA = False

# ________________________________________________________________________________________

if bvpAA is True:
    file = 'grow_beta_traj.mat'
    index = 2
    title = 'BVP $\Theta^{*}=(a,a)$'
    # title = 'Neural Network $\Theta^{*}=(a,a)$'
    special = 0
    theta1 = 1
    theta2 = 1

if bvpANA is True:
    file = 'data_train_a_na_1200.mat'
    file = 'data_test_a_na_200.mat'
    index = 2
    title = 'BVP $\Theta^{*}=(a,na)$'
    # title = 'Neural Network $\Theta^{*}=(a,a)$'
    special = 0
    theta1 = 1
    theta2 = 5

if bvpNAA is True:
    file = 'data_train_na_a_1200.mat'
    index = 2
    title = 'BVP $\Theta^{*}=(na,a)$'
    # title = 'Neural Network $\Theta^{*}=(a,a)$'
    special = 0
    theta1 = 5
    theta2 = 1

if bvpNANA is True:
    file = 'data_train_na_na_1200_corl.mat'
    index = 2
    title = 'BVP $\Theta^{*}=(na,na)$'
    # title = 'Neural Network $\Theta^{*}=(a,a)$'
    special = 0
    theta1 = 5
    theta2 = 5
# ____________________________________________________________________________________________________

data = scipy.io.loadmat(file)

font = {'family': 'serif', 'weight': 'bold', 'size': 20}
# matplotlib.rc('text', usetex = True)
matplotlib.rc('font', **font)
# params= {'text.latex.preamble' : [r'\boldmath']}
# plt.rcParams.update(params)

plt.rc('font', **font)

X = data['X']
V = data['V']
t = data['t']
# A = data['A']

data.update({'t0': data['t']})
idx0 = np.nonzero(np.equal(data.pop('t0'), 0.))[1]

fig, axs = plt.subplots(1, 1, figsize=(7, 7))

for n in range(1, len(idx0) + 1):
    if n == len(idx0):
        x1 = X[0, idx0[n - 1]:]
        x2 = X[index, idx0[n - 1]:]
        T = t[0, idx0[n - 1]:]
        V1 = V[0, idx0[n - 1]:]
        # A1 = A[0, idx0[n - 1]:]
        # axs.plot(T, V1)
        # axs.plot(T, A1)
        axs.plot(x1, x2, c='gray', alpha=0.2)  # 0.8
    # axs.plot(x1, x2, c='black')
    # axs.plot(x1, x2)

    else:
        x1 = X[0, idx0[n - 1]: idx0[n]]
        x2 = X[index, idx0[n - 1]: idx0[n]]
        T = t[0, idx0[n - 1]: idx0[n]]
        V1 = V[0, idx0[n - 1]: idx0[n]]
        # A1 = A[0, idx0[n - 1]: idx0[n]]
        # axs.plot(T, V1)
        # axs.plot(T, A1)
        axs.plot(x1, x2, c='gray', alpha=0.2)  # 0.8
    # axs.plot(x1, x2, c='black')
    # axs.plot(x1, x2)

train1 = patches.Rectangle((35 - theta1 * 0.75, 35 - 1 * 0.75), 3 + theta1 * 0.75 + 0.75,
                           3 + 1 * 0.75 + 0.75, linewidth=1, edgecolor='k', facecolor='none')
train2 = patches.Rectangle((35 - 1 * 0.75, 35 - theta2 * 0.75), 3 + 1 * 0.75 + 0.75,
                           3 + theta2 * 0.75 + 0.75, linewidth=1, edgecolor='k', facecolor='none')
start1 = patches.Rectangle((15, 15), 5, 5, linewidth=4, edgecolor='red', facecolor='none', zorder=3)
intersection1 = patches.Rectangle((34.25, 34.25), 4.5, 4.5, linewidth=1, edgecolor='grey', facecolor='grey')
na1 = patches.Rectangle((31.25, 34.25), 7.5, 4.5, linewidth=4, edgecolor='m', facecolor='none', zorder=3)
na2 = patches.Rectangle((34.25, 31.25), 4.5, 7.5, linewidth=4, edgecolor='blue', facecolor='none', zorder=3)
axs.add_patch(intersection1)
axs.add_patch(na1)
axs.add_patch(na2)
axs.add_patch(train1)
axs.add_patch(train2)
axs.add_patch(start1)
axs.set_xlim(15, 40)
# axs.set_xlabel('$d_1$: \\textbf{position of player 1}', fontweight='bold')
# axs.set_ylim(15, 40)
# axs.set_ylabel('$d_2$: \\textbf{position of player 2}', fontweight='bold')

axs.set_xlabel('$d_1$: position of player 1', fontweight='bold')
axs.set_ylim(15, 40)
axs.set_ylabel('$d_2$: position of player 2', fontweight='bold')
#
# axs.set_title(title)

arrowprops = dict(arrowstyle="->")

# plt.annotate('$\\textbf{player 2 yields to player 1}$', xy=(31, 24), xytext=(32.75, 20),
#              arrowprops=dict(facecolor='black', width=1.5), horizontalalignment='center', fontsize=14, color='black', fontweight='bold')
# plt.annotate('$\\textbf{player 1 yields to player 2}$', xy=(24, 31), xytext=(22, 37), arrowprops=dict(facecolor='black', width=1.5),
#              horizontalalignment='center', fontsize=14, color='black', fontweight='bold')

plt.annotate('player 2 yields to player 1', xy=(31, 24), xytext=(32.75, 20),
             arrowprops=dict(facecolor='black', width=1.5), horizontalalignment='center', fontsize=14, color='black', fontweight='bold')
plt.annotate('player 1 yields to player 2', xy=(24, 31), xytext=(22, 37), arrowprops=dict(facecolor='black', width=1.5),
             horizontalalignment='center', fontsize=14, color='black', fontweight='bold')
# plt.annotate('Equilibrium Interactions\n' + 'w/ Complete Info', xy=(35, 28), xytext=(32.5, 16), horizontalalignment='center',
# 			 fontweight='normal', fontsize=16, color='black')

# plt.show()
# fig.savefig('filename.png', dpi=1200)
plt.show()