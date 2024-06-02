import torch
from tqdm import tqdm
from torch.utils.data import random_split

from tools.utils import *
from evalMetrics import EvalMetrics

class Trainer:
    def __init__(self, model, optimizer, criterion, converter, opt,
                 train_dataset,
                 test_dataset):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        self.converter = converter
        self.device = next(self.model.parameters()).device
        self.opt = opt

        self.train_dataloader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=self.opt.batch_size,
                    shuffle=True)
        
        self.test_dataloader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=self.opt.batch_size,
                    shuffle=True)
        self.start_epoch = 0
        self.log = []

    def load_pretrained(self, path):
        checkpoint = torch.load(path, map_location=torch.device(self.device))
        print( checkpoint['desc'])

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch']
        self.log = checkpoint['log']

    def train(self, nepochs):
        for epoch in range(self.start_epoch + 1, self.start_epoch + nepochs + 1):
            # Train -----------------------
            avg_loss, avg_cer, avg_wer = self.epoch_train()
            print(f'Epoch: {epoch}/{self.start_epoch + nepochs}', end=" ")
            print('-->>> avg_loss = {:.4f} \t avg_cer = {:.2f} \t avg_wer = {:.2f}'.format(avg_loss,  avg_cer, avg_wer))
            self.log.append({
                'type': 'train',
                'epoch': epoch,
                'metric': {
                    'avg_loss': avg_loss,
                    'avg_cer': avg_cer,
                    'avg_wer': avg_wer
                }
            })

            # val ------------------------------
            if epoch % self.opt.valInterval == 0: 
                avg_loss,  avg_cer, avg_wer = self.epoch_eval()
                print(f"Tester.eval... {epoch}", end=" ")
                print('-->>> avg_loss = {:.4f} \t avg_cer = {:.2f} \t avg_wer = {:.2f}'.format(avg_loss,  avg_cer, avg_wer))
                self.log.append({
                    'type': 'val',
                    'epoch': epoch,
                    'metric': {
                        'avg_loss': avg_loss,
                        'avg_cer': avg_cer,
                        'avg_wer': avg_wer
                    }
                })

            # Save --------------------------
            if epoch % self.opt.saveInterval == 0:
                print('Saving Model...\n')

                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),  # Lưu trạng thái của mô hình
                    'optimizer_state_dict': self.optimizer.state_dict(),  # Lưu trạng thái của optimizer
                    'log': self.log
                }

                directory = self.opt.savedir 
                if not os.path.exists(directory):
                    os.makedirs(directory)
                    print(f"Đã tạo thư mục '{directory}'.")
                torch.save(checkpoint, directory +  f'/checkpoint-{epoch}.pth.tar')


    def epoch_train(self):
        self.model.train(True)
        avg_loss = 0
        evalMetrics = EvalMetrics()

        t = tqdm(iter(self.train_dataloader), total=len(self.train_dataloader))
        for _, (imgs, labels) in enumerate(t):
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
            # print(preds.shape)
            sim_preds = self.converter.decode(enc_preds.view(-1), preds_lengths, raw = False)
            # print(len(sim_preds), len(labels))
            # break
            evalMetrics.add(sim_preds, labels)
     
        avg_loss = avg_loss/len(self.train_dataloader)
        avg_cer, avg_wer = evalMetrics.eval()

        return avg_loss,  avg_cer, avg_wer
    
    def epoch_eval(self, print_ = True):
        self.model.eval()
        avg_loss = 0
        evalMetrics = EvalMetrics()

        with torch.no_grad():
            t = tqdm(iter(self.test_dataloader), total=len(self.test_dataloader))
            for batch_idx, (imgs, labels) in enumerate(t):
                imgs = imgs.to(self.device)
                preds = self.model(imgs)
                
                # Compute Loss -------------------------------------------
                preds_, preds_lengths, targets, target_lengths = GetInputCTCLoss(self.converter, preds, labels)
                loss = self.criterion(preds_.log_softmax(2), targets, preds_lengths, target_lengths) # ctc_loss chỉ dùng với cpu, dùng với gpu phức tạp hơn thì phải
                avg_loss += loss.detach().item()
                
                _, enc_preds = preds.max(2)
                sim_preds = self.converter.decode(enc_preds.view(-1), preds_lengths, raw = False)
                evalMetrics.add(sim_preds, labels)

        avg_loss = avg_loss/len(self.test_dataloader)
        avg_cer, avg_wer = evalMetrics.eval()

        # Display ----------------------------------------  
        if print_:
            raw_preds = self.converter.decode(enc_preds.view(-1), preds_lengths, raw = True)
            i = 5
            for raw_pred, pred, gt in zip(raw_preds, sim_preds, labels):
                print('='*30)
                print(f'raw: {raw_pred}')
                print(f'pred_text: {pred}', )
                print(f'gt: {gt}')
                i -= 1
                if( i == 0): break

        return avg_loss, avg_cer, avg_wer