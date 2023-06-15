import torch
from torch.optim import Adam
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import time
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lbfgsnew import LBFGSNew

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class OutputHook(list):
    """ Hook to capture module outputs.
    """
    def __call__(self, module, input, output):
        self.append(output)


def train(model, train_dataloader, val_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
          clip_grad=False, use_lbfgs=False):
    # optim = Adam(lr=lr, params=model.parameters())
    optim = torch.optim.SGD(lr=lr, params=model.parameters())
    scheduler = ReduceLROnPlateau(optim, mode='min', patience=2, verbose=True)



    if use_lbfgs:
        optim = LBFGSNew(model.parameters(), history_size=7, max_iter=2, line_search_fn=True, batch_mode=True)


    summaries_dir = os.path.join(model_dir, 'summaries')
    checkpoints_dir = os.path.join(model_dir, 'checkpoints_dir')
    if not os.path.exists(summaries_dir):
        os.makedirs(summaries_dir)

    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)

    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            model.train(True)
            if not epoch % epochs_til_checkpoint and epoch:
                checkpoint = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optim.state_dict()}
                torch.save(checkpoint, os.path.join(checkpoints_dir, 'model_epoch%04d.pth' % epoch))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.pth' % epoch),
                           np.array(train_losses))
            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()

                model_input = {key: value.to(device) for key, value in model_input.items()}
                gt = {key: value.to(device) for key, value in gt.items()}

                if use_lbfgs:
                    def closure():
                        if torch.is_grad_enabled():
                            optim.zero_grad()
                        model_output = model(model_input)
                        loss = loss_fn(model_output['model_out'], gt['values'])
                        loss *= loss # test
                        if loss.requires_grad:
                            loss.backward()
                        return loss
                    optim.step(closure)

                model_output = model(model_input)

                losses = loss_fn(model_output, gt)
                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()
                    writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss

                reg = 0.
                lambda_1 = 0.001
                for param in model.parameters():
                    reg += torch.norm(param, 1) ** 2

                train_loss += lambda_1 * reg

                train_losses.append(train_loss.item())
                writer.add_scalar("total_train_loss", train_loss, total_steps)



                # loss = loss_fn(model_output['model_out'], gt['values'])
                # loss *= 10  # for testing
                # train_losses.append(loss.item())


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


     

                pbar.update(1)
                if not total_steps % steps_til_summary:
                    tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f"
                               % (epoch, loss, time.time() - start_time))

                total_steps += 1

                # validation

            if not total_steps % 3:
                print("Checking Validation Loss: \n")
                model.eval()
                val_losses = []
                for step, (val_in, gt) in enumerate(val_dataloader):
                    val_input = {key: value.to(device) for key, value in val_in.items()}
                    gt = {key: value.to(device) for key, value in gt.items()}
                    model_output = model(val_input)
                    valLosses = loss_fn(model_output, gt)
                    val_loss = 0.
                    for loss_name, loss in valLosses.items():
                        single_val_loss = loss.mean()
                        writer.add_scalar(loss_name, single_val_loss, total_steps)
                        val_loss += single_val_loss

                    val_losses.append(val_loss.item())
                    writer.add_scalar("total_validation_loss", val_loss, total_steps)

                tqdm.write("Epoch %d, Total loss %0.6f, validation_loss %0.6f, iteration time %0.6f"
                           % (epoch, loss, (np.mean(val_losses)), time.time() - start_time))

                scheduler.step(np.mean(val_losses))

        torch.save(model.state_dict(), os.path.join(checkpoints_dir, 'model_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'), np.array(train_losses))

        # save checkpoint for training
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optim.state_dict()}
        torch.save(checkpoint, os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
