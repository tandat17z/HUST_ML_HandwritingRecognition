import torch
from tqdm import tqdm
from torch.utils.data import random_split

from utils.utils import *
from tester import Tester

class Trainer:
    def __init__(self, config, model, optimizer, criterion, dataset, converter):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])

        self.train_dataloader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=config.batch_size,
                    shuffle=True)
        self.test_dataloader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=config.batch_size,
                    shuffle=True)

        self.converter = converter
        self.device = next(self.model.parameters()).device

        self.tester = Tester(self.model, self.criterion, self.test_dataloader, self.converter)
        # self.losses = MetricTracker()
        # self.accs = MetricTracker()


    def train(self, num_epochs, valInterval, saveInterval):
        for epoch in range(1, num_epochs + 1):
            self.model.train(True)
            avg_loss, avg_levenshtein_loss = self._train_epoch(epoch)
            print('--> Epoch: [{}/{}]\t Avg Loss {:.4f} \t Avg Levenshtein Loss {:.2f}'.format(epoch, num_epochs, avg_loss, avg_levenshtein_loss))
            
            if epoch % valInterval == 0: 
                avg_loss, avg_levenshtein_loss = self.tester.eval()
                print('--> Val: \t Avg Loss {:.4f} \t Avg Levenshtein Loss {:.2f}'.format(epoch, avg_loss, avg_levenshtein_loss))
            
            if epoch % saveInterval == 0:
                print('Saving Model...')

                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),  # Lưu trạng thái của mô hình
                    'optimizer_state_dict': self.optimizer.state_dict(),  # Lưu trạng thái của optimizer
                }
                torch.save(checkpoint, 'pretrain/model_{epoch}.pth.tar')

    def _train_epoch(self, epoch_idx):
        # self.losses.reset()
        # self.accs.reset()
        avg_loss = 0
        avg_levenshtein_loss = 0

        t = tqdm(iter(self.train_dataloader), total=len(self.train_dataloader), desc='Epoch {}'.format(epoch_idx))
        for batch_idx, (imgs, labels) in enumerate(t):
            imgs = imgs.to(self.device)

            targets, target_lenghts = self.converter.encode(labels)
            targets = targets.to('cpu')
            target_lenghts = target_lenghts.to('cpu')

            self.optimizer.zero_grad()

            preds = self.model(imgs)
            
            b, l, c = preds.shape
            preds_ = preds.permute(1, 0, 2).to('cpu')
            preds_lengths = torch.full(size=(b,), fill_value=l, dtype=torch.long).to('cpu')

            loss = self.criterion(preds_.log_softmax(2), targets, preds_lengths, target_lenghts) # ctc_loss chỉ dùng với cpu, dùng với gpu phức tạp hơn thì phải
            loss.backward()
            self.optimizer.step()

            avg_loss += loss.detach().item()

            _, enc_preds = preds.max(2)
            sim_preds = self.converter.decode(enc_preds.view(-1), preds_lengths, raw = True)
            avg_levenshtein_loss += Levenshtein_loss(sim_preds, labels)

        return avg_loss/len(self.train_dataloader), avg_levenshtein_loss/len(self.train_dataloader)