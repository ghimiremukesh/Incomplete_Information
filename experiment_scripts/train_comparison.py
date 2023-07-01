import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dataio, icnn_pytorch_adaptive as icnn_pytorch, training_og as trainingInter
from torch.utils.data import DataLoader
import numpy as np
import torch
import configargparse

p = configargparse.ArgumentParser()
p.add_argument('--start', type=int, default=1,
               help='time-step to collect data')
opt = p.parse_args()

use_lbfgs = False

dt = 0.05

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

logging_root = f'logs/hexner_train_R2_timestep_{dt}'
save_root = f'hexner_test_case_R2_timestep_{dt}/'
# experiment_name = 'InitialTest'
if use_lbfgs:
    num_epochs = 5
else:
    num_epochs = 5  # 6

numpoints = 35000

start = opt.start
time = 0

model = icnn_pytorch.SingleBVPNet(in_features=6, out_features=1, type='relu', mode='mlp', hidden_features=128,
                                  num_hidden_layers=2, dropout=0)

model.to(device)

lr = 1e-4

t = np.arange(0, 1.1, dt)

ts = np.arange(dt, 1+dt, dt)

# mat_files = ['train_data_t_0.1.mat', 'train_data_t_0.2.mat', 'train_data_t_0.3.mat', 'train_data_t_0.4.mat',
#              'train_data_t_0.5.mat', 'train_data_t_0.6.mat', 'train_data_t_0.7.mat', 'train_data_t_0.8.mat',
#              'train_data_t_0.9.mat', 'train_data_t_1.0.mat']
# mat_files = ['train_data_t_0.2.mat', 'train_data_t_0.4.mat', 'train_data_t_0.6.mat',
#              'train_data_t_0.8.mat', 'train_data_t_1.0.mat']

mat_files = [f'train_data_t_{dt:.2f}.mat' for dt in ts]

for i in range(start, start + 1):  # just train t-dt
    assert i != 0, "no need to train final model, we know the value!"
    start_epoch = num_epochs - 1
    print(f'\n Training for timestep: t = {t[i]}\n')
    # dataset = dataio.TrainInterTimeGrad(os.path.join(save_root, matfiles[i-1]))
    dataset = dataio.TrainInterTime(os.path.join(save_root, mat_files[i - 1]))
    train_set, val_set = torch.utils.data.random_split(dataset, [450000, 50000])
    train_dataloader = DataLoader(train_set, shuffle=True, batch_size=128, pin_memory=True, num_workers=0)
    val_dataloader = DataLoader(val_set, shuffle=False, batch_size=128, pin_memory=True, num_workers=0)
    loss_fn = torch.nn.L1Loss(reduction='mean')
    root_path = os.path.join(logging_root, f't_{i}/')

    trainingInter.train(model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                        epochs=num_epochs, lr=lr, steps_til_summary=1000, epochs_til_checkpoint=1,
                        model_dir=root_path, loss_fn=loss_fn, clip_grad=False, use_lbfgs=use_lbfgs)
