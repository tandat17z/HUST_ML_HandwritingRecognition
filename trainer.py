import torch
from tester import Tester
from utils.utils import MetricTracker
from tqdm import tqdm

class Trainer:
    def __init__(self, model, optimizer, criterion, dataloader):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloader = dataloader
        self.device = next(self.model.parameters()).device

        self.losses = MetricTracker()
        self.accs = MetricTracker()

        self.tester = Tester(self.config, self.model)

    def train(self, num_epochs, valInterval, saveInterval):
        for epoch in range(1, num_epochs + 1):
            self.model.train(True)
            result = self._train_epoch(epoch)
            print('Epoch: [{0}]\t Avg Loss {loss:.4f}\t Avg Accuracy {acc:.3f}'.format(epoch, loss=result['loss'], acc=result['acc']))
            
            if epoch % valInterval == 0: 
                self.tester.eval()

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
        for batch_idx, (img, label) in enumerate(t):
            self.optimizer.zero_grad()

            preds = self.model(img)
            
            loss = self.criterion(preds, label)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.detach().item()

            # _, preds = preds.max(2)
            # preds = preds.transpose(1, 0).contiguous().view(-1)
            # sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
            # cer_loss = utils.cer_loss(sim_preds, cpu_texts)
            # return cost, cer_loss, len(cpu_images)
        return running_loss