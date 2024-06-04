from tqdm import tqdm
import torch
from tools import utils

class Tester:
    def __init__(self, model, criterion, converter, config = {'imgW': 768, 'threshold': 30}):
        self.model = model
        self.converter = converter
        self.criterion = criterion
        self.device = next(self.model.parameters()).device
        self.config = config

    def setDataset(self, dataset, batch_size = 64):
        self.dataloader = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=batch_size,
                        shuffle=True)
        self.batch_size = batch_size

    def predict(self, imgpath):
        self.model.eval()
        # img = utils.img_loader(imgpath, imgW = 800, scale=1.5)
        img = utils.img_loader(imgpath, imgW = self.config['imgW'], threshold=self.config['threshold'])
        input = img.unsqueeze(0).to(self.device)

        pred = self.model(input)
                
        b, l, c = pred.shape
        pred_ = pred.permute(1, 0, 2).to('cpu')
        pred_lengths = torch.full(size=(b,), fill_value=l, dtype=torch.long).to('cpu')

        _, enc_pred = pred_.max(2)
        sim_pred = self.converter.decode(enc_pred.view(-1), pred_lengths, raw = False)
        return sim_pred