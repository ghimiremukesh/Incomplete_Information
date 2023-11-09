import torch
import numpy as np
import icnn_pytorch_adaptive as icnn
import training_og as training
import configargparse
from torch.utils.data import DataLoader
import dataio
import os

# training script for soccer
p = configargparse.ArgumentParser()
p.add_argument('--start', type=int, default=1,
               help='time-step to collect data')
p.add_argument('--validation', type=bool, default=True, help='check validation loss')
opt = p.parse_args()

logging_root = f'logs/soccer_velocity'
save_root = f'soccer_velocity/'


model = icnn.SingleBVPNet(in_features=5, out_features=1, num_hidden_layers=2, hidden_features=32, mode='mlp',
                          type='relu')
start = opt.start
dt = 0.1
t = np.arange(0, 1.1, dt)

ts = np.arange(dt, 1+dt, dt)

num_epochs = 10
lr = 1e-3

mat_files = [f'train_data_t_{dt:.2f}.mat' for dt in ts]

for i in range(start, start + 1):  # just train t-dt
    assert i != 0, "no need to train final model, we know the value!"
    start_epoch = num_epochs - 1
    print(f'\n Training for timestep: t = {t[i]}\n')

    dataset = dataio.TrainInterTime(os.path.join(save_root, mat_files[i - 1]))
    # train_set, val_set = torch.utils.data.random_split(dataset, [90000, 10000])
    train_dataloader = DataLoader(dataset, shuffle=True, batch_size=128, pin_memory=True, num_workers=0)
    # val_dataloader = DataLoader(val_set, shuffle=False, pin_memory=True, num_workers=0)
    loss_fn = torch.nn.MSELoss(reduction='sum')
    root_path = os.path.join(logging_root, f't_{i}/')

    opt.validation = False
    val_dataloader = None
    training.train(model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                        epochs=num_epochs, lr=lr, steps_til_summary=100, epochs_til_checkpoint=10,
                        model_dir=root_path, loss_fn=loss_fn, clip_grad=False, use_lbfgs=False,
                        validation=opt.validation)



