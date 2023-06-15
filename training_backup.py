import torch
import utils
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
import shutil
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lbfgsnew import LBFGSNew

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
          val_fn=None, load_dir=None, start_epoch=0, clip_grad=False, num_iterations=1000, use_lbfgs=False):
    optim = torch.optim.Adam(lr=lr, params=model.parameters())
    # optim = torch.optim.SGD(lr=lr, params=model.parameters())
    scheduler = ReduceLROnPlateau(optim, mode='min', patience=500, verbose=True)

    if use_lbfgs:
        optim = LBFGSNew(params=model.parameters(), history_size=7, max_iter=4, line_search_fn=True, batch_mode=False)

    # do not load previously trained model # uncomment to load

    if start_epoch > 0:
        model_path = os.path.join(load_dir, 'checkpoints', 'model_epoch_%04d.pth' % start_epoch)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        model.train()
        optim.load_state_dict(checkpoint['optimizer'])
        optim.param_groups[0]['lr'] = lr
        assert (start_epoch == checkpoint['epoch'])
    else:
        if os.path.exists(model_dir):
            val = input("The model directory %s exists. Overrite? (y/n)" % model_dir)
            if val == 'y':
                shutil.rmtree(model_dir)
        os.makedirs(model_dir)

    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)

    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs * num_iterations) as pbar:
        train_losses = []
        for epoch in range(epochs):
            if not epoch % epochs_til_checkpoint and epoch:
                checkpoint = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optim.state_dict()}
                torch.save(checkpoint, os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch), np.array(train_losses))

            for step, (model_input, gt) in enumerate(train_dataloader):
                for _ in range(num_iterations):
                    start_time = time.time()

                    model_input = {key: value.to(device) for key, value in model_input.items()}
                    gt = {key: value.to(device) for key, value in gt.items()}

                    if use_lbfgs:
                        def closure():
                            if torch.is_grad_enabled():
                                optim.zero_grad()   
                            model_output = model(model_input)
                            losses = loss_fn(model_output, gt, val_fn)
                            train_loss = 0.
                            for loss_name, loss in losses.items():
                                train_loss += loss.mean()
                            
                            if train_loss.requires_grad:
                                train_loss.backward()
                            return train_loss
                        optim.step(closure)

                    model_output = model(model_input)
                    losses = loss_fn(model_output, gt, val_fn)

                    train_loss = 0.
                    for loss_name, loss in losses.items():
                        single_loss = loss.mean()

                        writer.add_scalar(loss_name, single_loss, total_steps)
                        train_loss += single_loss

                    train_losses.append(train_loss.item())
                    writer.add_scalar("total_train_loss", train_loss, total_steps)

                    if not total_steps % steps_til_summary:
                        torch.save(model.state_dict(), os.path.join(checkpoints_dir, 'model_current.pth'))

                    if not use_lbfgs:
                        optim.zero_grad()
                        train_loss.backward()

                        if clip_grad:
                            if isinstance(clip_grad, bool):
                                torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1.)
                            else:
                                torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=clip_grad)

                        optim.step()
                    
                    model.convexify()
                    # scheduler.step(train_loss)

                    pbar.update(1)

                    if not total_steps % steps_til_summary:
                        tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss,
                                                                                         time.time() - start_time))
                    total_steps += 1

        torch.save(model.state_dict(), os.path.join(checkpoints_dir, 'model_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))

        # save checkpoint for training
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optim.state_dict()}
        torch.save(checkpoint, os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
