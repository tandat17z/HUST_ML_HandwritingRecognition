from tqdm import tqdm
import torch
from utils import utils

class Tester:
    def __init__(self, model, criterion, converter):
        self.model = model
        self.converter = converter
        self.criterion = criterion
        self.device = next(self.model.parameters()).device
        
    def setDataset(self, dataset, batch_size = 64):
        self.dataloader = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=batch_size,
                        shuffle=True)
        self.batch_size = batch_size

    def predict(self, imgpath):
        self.model.eval()
        # img = utils.img_loader(imgpath, imgW = 800, scale=1.5)
        img = utils.img_loader(imgpath, imgW = 768, threshold=0)
        input = img.unsqueeze(0).to(self.device)

        pred = self.model(input)
                
        b, l, c = pred.shape
        pred_ = pred.permute(1, 0, 2).to('cpu')
        pred_lengths = torch.full(size=(b,), fill_value=l, dtype=torch.long).to('cpu')

        _, enc_pred = pred_.max(2)
        sim_pred = self.converter.decode(enc_pred.view(-1), pred_lengths, raw = False)
        return sim_pred

    # def eval(self, print_ = True):
    #     print("Tester.eval...")
    #     assert self.dataloader != False, "Tester: error self.dataloader = False "
        
    #     self.model.eval()
    #     with torch.no_grad():
    #         total_loss = 0
    #         levenshtein_loss = 0

    #         t = tqdm(iter(self.dataloader), total=len(self.dataloader))
    #         for batch_idx, (imgs, labels) in enumerate(t):
    #             imgs = imgs.to(self.device)

    #             targets, target_lenghts = self.converter.encode(labels)
    #             targets = targets.to('cpu')
    #             target_lenghts = target_lenghts.to('cpu')

    #             preds = self.model(imgs)
                
    #             # Compute Loss -------------------------------
    #             b, l, c = preds.shape
    #             preds_ = preds.permute(1, 0, 2).to('cpu')
    #             preds_lengths = torch.full(size=(b,), fill_value=l, dtype=torch.long).to('cpu')

    #             loss = self.criterion(preds_.log_softmax(2), targets, preds_lengths, target_lenghts) # ctc_loss chỉ dùng với cpu, dùng với gpu phức tạp hơn thì phải
    #             total_loss += loss.detach().item()

    #             _, enc_preds = preds.max(2)
    #             sim_preds = self.converter.decode(enc_preds.view(-1), preds_lengths, raw = False)
    #             levenshtein_loss += self.converter.Levenshtein_loss(sim_preds, labels)

    #     # Display ----------------------------------------  
    #     if print_:
    #         raw_preds = self.converter.decode(enc_preds.view(-1), preds_lengths, raw = True)
    #         i = 5
    #         for raw_pred, pred, gt in zip(raw_preds, sim_preds, labels):
    #             print('='*30)
    #             print(f'raw: {raw_pred}')
    #             print(f'pred_text: {pred}', )
    #             print(f'gt: {gt}')
    #             i -= 1
    #             if( i == 0): break

    #     total_loss = total_loss/self.dataloader.sampler.num_samples * self.batch_size
    #     levenshtein_loss = levenshtein_loss/self.dataloader.sampler.num_samples

    #     return total_loss, levenshtein_loss