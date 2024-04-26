import torch
from utils import utils

class DatasetImg(torch.utils.data.Dataset):
    def __init__(self, imgFolder, labelFolder):
        self.imgH = 32
        self.imgW = 512
        self.imlist = utils.flist_reader(imgFolder, labelFolder)

    def __getitem__(self, index):
        imgpath, imglabel = self.imlist[index]

        img = utils.img_loader(imgpath, self.imgH, self.imgW)
        target = utils.target_loader(imglabel)

        return img, target

    def __len__(self):
        return len(self.imlist)