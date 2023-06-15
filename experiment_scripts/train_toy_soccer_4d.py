from pickle import FALSE
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataio, icnn_pytorch_adaptive as icnn_pytorch, loss_functions, training_backup as training, training_og as trainingInter
from torch.utils.data import DataLoader
import numpy as np
import torch
import configargparse

p = configargparse.ArgumentParser()
p.add_argument('--start', type=int, default=4,
               help='time-step to collect data')
opt = p.parse_args()

use_lbfgs = False

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

logging_root = 'logs/train_new_case_follow_zero_relative'
save_root = 'new_case_data_follow_zero_relative'
experiment_name = 'InitialTest'
if use_lbfgs:
    num_epochs = 5
else:
    num_epochs = 5 #6

numpoints = 35000

start = opt.start
time = 0



model = icnn_pytorch.SingleBVPNet(in_features=5, out_features=1, type='relu', mode='mlp', hidden_features=128,
                                   num_hidden_layers=2, dropout=0)
#model = modules.SingleBVPNet(in_features=4, out_features=1, type='relu', mode='mlp', hidden_features=64,
#                             num_hidden_layers=2)


model.to(device)


dt = 0.1

lr = 1e-3

t = np.arange(0, 1.1, dt)

matfiles = ['train_data_t_0.1.mat', 'train_data_t_0.2.mat', 'train_data_t_0.3.mat', 'train_data_t_0.4.mat', 'train_data_t_0.5.mat', 'train_data_t_0.6.mat', 'train_data_t_0.7.mat','train_data_t_0.8.mat','train_data_t_0.9.mat','train_data_t_1.0.mat']
# matfiles = ['train_data_t_0.1.mat', 'train_data_t_1.mat']

for i in range(start, start+1):   # just train t-dt
    if i == 0:
        lr = 2e-5
        load_dir = None
        start_epoch = 0
        val_model = None
        root_path = os.path.join(logging_root, f't_{i}/')
        print(f'\n Training for timestep: t = {t[i]}\n')
        dataset = dataio.OneSidedGame_4d(numpoints=numpoints, t=t[i], dt=dt, u_max=1, d_max=0.8, num_src_samples=1)
        dataloader = DataLoader(dataset, shuffle=True, batch_size=4, pin_memory=True, num_workers=0)
        loss_fn = loss_functions.initialize_one_sided_game(dataset)
        #
        training.train(model=model, train_dataloader=dataloader, epochs=num_epochs, lr=lr, steps_til_summary=100,
                       epochs_til_checkpoint=10, model_dir=root_path, loss_fn=loss_fn, val_fn=val_model,
                       load_dir=load_dir, start_epoch=start_epoch, clip_grad=True, num_iterations=1000, use_lbfgs=use_lbfgs)
    else:
        # load_dir_pth = 'logs/t_0/checkpoints/model_epoch_0019.pth'
        # load_dir = os.path.join(logging_root, f't_{i-1}/')
        start_epoch = 4 #num_epochs - 1

        # val_model = icnn_pytorch.SingleBVPNet(in_features=4, out_features=1, type='relu', mode='mlp',
        #                                       hidden_features=32, num_hidden_layers=3)
        # val_model.to(device)
        # # model_path = os.path.join(load_dir, 'checkpoints', 'model_epoch_%04d.pth' % start_epoch)
        # model_path = load_dir_pth
        # checkpoint = torch.load(model_path, map_location=device)
        # val_model.load_state_dict(checkpoint['model'])
        # val_model.eval()

        print(f'\n Training for timestep: t = {t[i]}\n')
        # dataset = dataio.TrainInterTimeGrad(os.path.join(save_root, matfiles[i-1]))
        dataset = dataio.TrainInterTime(os.path.join(save_root, matfiles[i-1]))
        train_set, val_set = torch.utils.data.random_split(dataset, [900000, 100000])
        train_dataloader = DataLoader(train_set, shuffle=True, batch_size=128, pin_memory=True, num_workers=0)
        val_dataloader = DataLoader(val_set, shuffle=False, batch_size=128, pin_memory=True, num_workers=0)
        loss_fn = torch.nn.L1Loss(reduction='mean')
        # loss_fn = loss_functions.initialize_val_grad_loss(dataset)
        # loss_fn = torch.nn.HuberLoss(reduction='sum')
        root_path = os.path.join(logging_root, f't_{i}/')

        trainingInter.train(model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                            epochs=num_epochs, lr=lr, steps_til_summary=1000, epochs_til_checkpoint=1,
                            model_dir=root_path, loss_fn=loss_fn, clip_grad=False, use_lbfgs=False)

    # root_path = os.path.join(logging_root, f't_{i}/')
    # print(f'\n Training for timestep: t = {t[i]}\n')
    # dataset = dataio.OneSidedGame(numpoints=numpoints, t=t[i], dt=0.5, u_max=1, d_max=0.8, num_src_samples=1)
    # dataloader = DataLoader(dataset, shuffle=True, batch_size=4, pin_memory=True, num_workers=0)
    # loss_fn = loss_functions.initialize_one_sided_game(dataset)
    #
    # training.train(model=model, train_dataloader=dataloader, epochs=num_epochs, lr=lr, steps_til_summary=100,
    #                epochs_til_checkpoint=1000, model_dir=root_path, loss_fn=loss_fn, val_fn=val_model,
    #                load_dir=load_dir, start_epoch=start_epoch, clip_grad=False, num_iterations=1000)


# # test
# x = dataset.__getitem__(0)[0]['coords']
# loss_fn.dynamics(x, 1, 0.8)

