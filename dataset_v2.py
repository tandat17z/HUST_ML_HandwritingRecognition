import torch
from utils import utils

class DatasetImg_v2(torch.utils.data.Dataset):
    def __init__(self, imgFolder, labelFolder, imgH = 32, imgW = 800, scale = 1):
        self.imgH = imgH
        self.imgW = imgW
        self.scale = scale
        self.imlist = utils.flist_reader(imgFolder, labelFolder)

    def __getitem__(self, index):
        idx = index % len(self.imlist)
        imgpath, imglabel = self.imlist[idx]

        if index < len(self.imlist):
            img = utils.img_loader(imgpath, self.imgH, self.imgW, self.scale, alignment='left')
        elif index < 2*len(self.imlist):
            img = utils.img_loader(imgpath, self.imgH, self.imgW, self.scale, alignment='center')
        else:
            img = utils.img_loader(imgpath, self.imgH, self.imgW, self.scale, alignment='right')
        
        target = utils.target_loader(imglabel)
        return img, target


    def __len__(self):
        return len(self.imlist) * 3