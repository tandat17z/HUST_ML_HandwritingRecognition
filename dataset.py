import torch
from utils import utils

class DatasetImg(torch.utils.data.Dataset):
    def __init__(self, imgFolder, labelFolder, imgH = 32, imgW = 512):
        self.imgH = imgH
        self.imgW = imgW
        self.imlist = utils.flist_reader(imgFolder, labelFolder)

    def __getitem__(self, index):
        imgpath, imglabel = self.imlist[index]

        img = utils.img_loader(imgpath, self.imgH, self.imgW, threshold = 0)
        target = utils.target_loader(imglabel)

        return img, target

    def __len__(self):
        return len(self.imlist)