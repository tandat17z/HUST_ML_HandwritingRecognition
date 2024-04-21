import torch
from tqdm import tqdm
from utils.utils import *

class Trainer:
    def __init__(self, model, optimizer, criterion, dataloader, converter):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloader = dataloader

        self.converter = converter
        self.device = next(self.model.parameters()).device

        # self.losses = MetricTracker()
        # self.accs = MetricTracker()

        # self.tester = Tester(self.config, self.model)

    def train(self, num_epochs, valInterval, saveInterval):
        for epoch in range(1, num_epochs + 1):
            self.model.train(True)
            running_loss = self._train_epoch(epoch)
            print('--> Epoch: [{0}]\t Avg Loss {:.4f}\t Avg Accuracy {:.3f}'.format(epoch, running_loss, running_loss))
            
            # if epoch % valInterval == 0: 
            #     self.tester.eval()

            if epoch % saveInterval == 0:
                print('Saving Model...')
                torch.save({
                    'epoch': epoch,
                    'model': self.model,
                    'optimizer': self.optimizer,
                }, 'pretrain/model_{epoch}.pth.tar')

    def _train_epoch(self, epoch_idx):
        # self.losses.reset()
        # self.accs.reset()
        running_loss = 0
        t = tqdm(iter(self.dataloader), total=len(self.dataloader), desc='Epoch {}'.format(epoch_idx))
        for batch_idx, (imgs, labels) in enumerate(t):
            imgs = imgs.to(self.device)

            targets, target_lenghts = self.converter.encode(labels)
            targets = targets.to(self.device)
            target_lenghts = target_lenghts.to(self.device)

            self.optimizer.zero_grad()

            preds = self.model(imgs)
            
            b, l, c = preds.shape
            preds = preds.permute(1, 0, 2)
            preds_lengths = torch.full(size=(b), fill_value=l, dtype=torch.long).to(self.device)

            loss = self.criterion(preds, targets, preds_lengths, target_lenghts)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.detach().item()

        return running_loss