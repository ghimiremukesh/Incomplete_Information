import matplotlib.pyplot as plt

f = open('experiment_scripts/logs/tests/checkpoints/train_losses_final.txt', 'r').readlines()
N = len(f)
print(N)
train_err = []

for i in range(0, N):
    train_err.append(float(f[i].split(' ')[0]))  # please change the index, it might not be 0

x = [z for z in range(0, N)]
font = {'family': 'normal', 'weight': 'normal', 'size': 16}
plt.rc('font', **font)
fig, axs = plt.subplots(1, 1, figsize=(8, 6))


axs.set_xlabel('epochs')
axs.set_yscale('log')
# axs.set_yticks([0, 1000, 2000, 3000])
axs.plot(x, train_err, 'g-', label='train loss')
plt.legend()

title = 'Training Loss with PICNN'
axs.set_title(title, fontsize=20)
plt.show()
