import torch
from tools import utils

class DatasetImg(torch.utils.data.Dataset):
    def __init__(self, imgFolder, labelFolder, imgH = 32, imgW = 768, threshold = 30):
        self.imgH = imgH
        self.imgW = imgW
        self.threshold = threshold
        self.imlist = utils.flist_reader(imgFolder, labelFolder)

    def __getitem__(self, index):
        idx = index % len(self.imlist)
        imgpath, imglabel = self.imlist[idx]

        if index < len(self.imlist):
            img = utils.img_loader(imgpath, self.imgH, self.imgW, alignment='left', threshold=self.threshold)
        elif index < 2*len(self.imlist):
            img = utils.img_loader(imgpath, self.imgH, self.imgW, alignment='center', threshold=self.threshold)
        else:
            img = utils.img_loader(imgpath, self.imgH, self.imgW, alignment='right', threshold=self.threshold)
        
        target = utils.target_loader(imglabel)
        return img, target


    def __len__(self):
        return len(self.imlist) * 3