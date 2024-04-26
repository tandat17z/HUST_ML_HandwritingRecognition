import torch
from tqdm import tqdm
from torch.utils.data import random_split

from utils.utils import *
from tester import Tester

class Trainer:
    def __init__(self, model, optimizer, criterion, converter, dataset, batch_size):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.converter = converter
        self.device = next(self.model.parameters()).device
        self.batch_size = batch_size

        train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])
        self.train_dataloader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=self.batch_size,
                    shuffle=True)

        self.tester = Tester(self.model, 
                            criterion=self.criterion, 
                            converter=self.converter,
                            dataset=test_dataset,
                            batch_size=self.batch_size)


    def train(self, num_epochs, valInterval, saveInterval, start_epoch = 0):
        for epoch in range(start_epoch + 1, start_epoch + num_epochs + 1):
            # Train -------------------------
            self.model.train(True)
            total_loss, levenshtein_loss = self._train_epoch(epoch)
            print('Epoch: [{}/{}]\t avg_Loss = {:.4f} \t Levenshtein Loss per 1 sentence = {:.2f}'.format(epoch, start_epoch + num_epochs, total_loss, levenshtein_loss))
            
            # Val ---------------------------
            if epoch % valInterval == 0: 
                total_loss, levenshtein_loss = self.tester.eval()
                print('--> Val: \t avg_Loss = {:.4f} \t Levenshtein Loss per 1 sentence = {:.2f}'.format(total_loss, levenshtein_loss))
            
            # Save --------------------------
            if epoch % saveInterval == 0:
                print('Saving Model...\n')

                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),  # Lưu trạng thái của mô hình
                    'optimizer_state_dict': self.optimizer.state_dict(),  # Lưu trạng thái của optimizer
                }
                torch.save(checkpoint, f'pretrain/model_{epoch}.pth.tar')

    def _train_epoch(self, epoch_idx):
        # self.losses.reset()
        # self.accs.reset()
        total_loss = 0
        levenshtein_loss = 0

        t = tqdm(iter(self.train_dataloader), total=len(self.train_dataloader), desc='Epoch {}'.format(epoch_idx))
        for batch_idx, (imgs, labels) in enumerate(t):
            imgs = imgs.to(self.device)

            targets, target_lenghts = self.converter.encode(labels)
            targets = targets.to('cpu')
            target_lenghts = target_lenghts.to('cpu')

            self.optimizer.zero_grad()

            preds = self.model(imgs)
            
            # Compute Loss -------------------------------------------
            b, l, c = preds.shape
            preds_ = preds.permute(1, 0, 2).to('cpu')
            preds_lengths = torch.full(size=(b,), fill_value=l, dtype=torch.long).to('cpu')

            loss = self.criterion(preds_.log_softmax(2), targets, preds_lengths, target_lenghts) # ctc_loss chỉ dùng với cpu, dùng với gpu phức tạp hơn thì phải
            assert (not torch.isnan(loss) and not torch.isinf(loss)), "Loss value is NaN or Inf"
            
            # Backward -------------------------------------------------
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.detach().item()

            _, enc_preds = preds.max(2)
            sim_preds = self.converter.decode(enc_preds.view(-1), preds_lengths, raw = False)
            levenshtein_loss += self.converter.Levenshtein_loss(sim_preds, labels)

        total_loss = total_loss/self.train_dataloader.sampler.num_samples * self.batch_size
        levenshtein_loss = levenshtein_loss/self.train_dataloader.sampler.num_samples

        return total_loss, levenshtein_loss