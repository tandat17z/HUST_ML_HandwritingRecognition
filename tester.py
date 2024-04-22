from utils.utils import *
from tqdm import tqdm

class Tester:
    def __init__(self, model, criterion, dataloader, converter):
        self.model = model
        self.device = next(self.model.parameters()).device
        self.dataloader = dataloader
        self.converter = converter
        self.criterion = criterion

    def eval(self, print_ = True):
        print("Tester.eval...")
        self.model.eval()
        with torch.no_grad():
            avg_loss = 0
            avg_levenshtein_loss = 0

            t = tqdm(iter(self.dataloader), total=len(self.dataloader))
            for batch_idx, (imgs, labels) in enumerate(t):
                imgs = imgs.to(self.device)

                targets, target_lenghts = self.converter.encode(labels)
                targets = targets.to('cpu')
                target_lenghts = target_lenghts.to('cpu')

                preds = self.model(imgs)
                
                b, l, c = preds.shape
                preds_ = preds.permute(1, 0, 2).to('cpu')
                preds_lengths = torch.full(size=(b,), fill_value=l, dtype=torch.long).to('cpu')

                loss = self.criterion(preds_.log_softmax(2), targets, preds_lengths, target_lenghts) # ctc_loss chỉ dùng với cpu, dùng với gpu phức tạp hơn thì phải
            
                avg_loss += loss.detach().item()
                _, enc_preds = preds.max(2)
                sim_preds = self.converter.decode(enc_preds.view(-1), preds_lengths, raw = False)
                avg_levenshtein_loss += Levenshtein_loss(sim_preds, labels)
                
        if print_:
            raw_preds = self.converter.decode(enc_preds.view(-1), preds_lengths, raw = True)
            i = 5
            for raw_pred, pred, gt in zip(raw_preds, sim_preds, labels):
                print('='*30)
                print('raw: {}', raw_pred)
                print('pred_text: {}', pred)
                print('gt: {}', gt)
                i -= 1
                if( i == 0): break
        return avg_loss/len(self.dataloader), avg_levenshtein_loss/len(self.dataloader)