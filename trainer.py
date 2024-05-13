import torch
from tqdm import tqdm
from torch.utils.data import random_split

from utils.utils import *
from tester import Tester

class Trainer:
    def __init__(self, model, optimizer, criterion, converter, opt):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        self.converter = converter
        self.device = next(self.model.parameters()).device
        self.batch_size = batch_size

        self.train_dataloader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=self.batch_size,
                    shuffle=True)
        self.test_dataloader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=self.batch_size,
                    shuffle=True)
        self.start_epoch = 0

    def load_pretrained(self, path):
        self.checkpoint_path = path
        checkpoint = torch.load(path, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch']
        self.log = checkpoint['log']

    def train(self, nepochs):
        for epoch in range(self.start_epoch + 1, self.start_epoch + nepochs + 1):
            # Train -----------------------
            print(f'Epoch: {epoch}/{self.start_epoch + nepochs}')
            avg_loss, avg_levenshtein_loss = self.train()
            print('-->>> avg_loss/batch = {:.4f} \t avg_levenshtein_loss/sentence = {:.2f}'.format(avg_loss, avg_levenshtein_loss))
            self.log.append({
                'type': 'train',
                'epoch': epoch,
                'avg_Loss': avg_loss,
                'levenshtein_Loss': avg_levenshtein_loss
            })

            # val ------------------------------
            if epoch % opt.valInterval == 0: 
                print("Tester.eval...")


    def train(self):
        self.model.train(True)
        avg_loss = 0
        avg_levenshtein_loss = 0

        t = tqdm(iter(self.train_dataloader), total=len(self.train_dataloader))
        for batch_idx, (imgs, labels) in enumerate(t):
            imgs = imgs.to(self.device)
            self.optimizer.zero_grad()
            preds = self.model(imgs)
            
            # Compute Loss -------------------------------------------
            preds_, preds_lengths, targets, target_lengths = GetInputCTCLoss(self.converter, preds, labels)
            loss = self.criterion(preds_.log_softmax(2), targets, preds_lengths, target_lengths) # ctc_loss chỉ dùng với cpu, dùng với gpu phức tạp hơn thì phải
            assert (not torch.isnan(loss) and not torch.isinf(loss)), "Loss value is NaN or Inf"
            
            # Backward -------------------------------------------------
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            loss.backward()
            self.optimizer.step()

            avg_loss += loss.detach().item()

            _, enc_preds = preds.max(2)
            sim_preds = self.converter.decode(enc_preds.view(-1), preds_lengths, raw = False)
            avg_levenshtein_loss += self.converter.Levenshtein_loss(sim_preds, labels)

        avg_loss = avg_loss/self.train_dataloader.sampler.num_samples*self.batch_size
        avg_levenshtein_loss = avg_levenshtein_loss/self.train_dataloader.sampler.num_samples

        return avg_loss, avg_levenshtein_loss