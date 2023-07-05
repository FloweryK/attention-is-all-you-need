import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class Trainer:
    def __init__(self, config, model, criterion, optimizer):
        self.model = model.to(config.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.writer = SummaryWriter()
    
    def run_epoch(self, epoch, dataloader, device, train=True):
        losses = []
        match = []

        if train:
            self.model.train()
        else:
            self.model.eval()

        with tqdm(total=len(dataloader), desc=f"{'Train' if train else 'Test'} {epoch}") as pbar:
            for data in dataloader:
                # load input, label
                label, x_enc, x_dec = (x.to(device) for x in data)

                # predict
                if train:
                    self.optimizer.zero_grad()
                predict = self.model(x_enc, x_dec)

                # calculate loss
                loss = self.criterion(predict, label)
                loss_val = loss.item()
                losses.append(loss_val)

                # update model
                if train:
                    loss.backward()
                    self.optimizer.step()    
                
                # calculate performance
                match.extend(torch.eq(torch.argmax(predict, dim=1), label).detach().cpu().tolist())
                accuracy = np.sum(match) / len(match) if match else 0
            
                # update progress bar
                pbar.update(1)
                pbar.set_postfix_str(f"Loss: {loss_val:.3f} ({np.mean(losses):.3f}) | Acc: {accuracy:.3f}")
            
        # tensorboard
        self.writer.add_scalar('Loss/train' if train else 'Loss/test', np.mean(losses), epoch)
        self.writer.add_scalar('Acc/train' if train else 'Acc/test', accuracy, epoch)

