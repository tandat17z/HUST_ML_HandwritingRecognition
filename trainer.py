import torch
from tqdm import tqdm
from torch.utils.data import random_split

from utils.utils import *
from tester import Tester

class Trainer:
    def __init__(self, model, optimizer, criterion, converter, train_dataset, batch_size):
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

    def train(self):
        # self.losses.reset()
        # self.accs.reset()
        total_loss = 0
        levenshtein_loss = 0

        t = tqdm(iter(self.train_dataloader), total=len(self.train_dataloader))
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