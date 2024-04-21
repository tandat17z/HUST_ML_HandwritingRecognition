import torch
import os
from PIL import Image
import utils
from torchvision.transforms import ToTensor
import numpy as np

class DatasetImg(torch.utils.data.Dataset):
    def __init__(self, imgFolder, labelFolder, imgW, imgH):
        self.imgW = imgW
        self.imgH = imgH
        self.imlist = flist_reader(imgFolder, labelFolder)

    def __getitem__(self, index):
        imgpath, imglabel = self.imlist[index]

        img = self.img_loader(imgpath)
        target = self.target_loader(imglabel)

        return img, target

    def __len__(self):
        return len(self.imlist)
    
    def img_loader(self, path):
        img = Image.open(path).convert('L')
        img = img.point(lambda p: 255 - p) # chuyển background về màu đen 0

        # Cắt bỏ khoảng trống bị thừa xung quanh
        img_array = np.array(img)
        non_empty_columns = np.where(img_array.max(axis=0) > 0)[0]
        non_empty_rows = np.where(img_array.max(axis=1) > 0)[0]
        cropped_img = img_array[:, min(non_empty_columns):max(non_empty_columns) + 1]
        img = Image.fromarray(cropped_img)

        # Resize hình ảnh + thêm padding (nếu cần)
        desired_w, desired_h = self.imgW, self.imgH #(width, height)
        img_w, img_h = img.size  # old_size[0] is in (width, height) format
        ratio = 1.0*img_w/img_h
        new_w = int(desired_h*ratio)
        new_w = new_w if desired_w == None else min(desired_w, new_w)
        img = img.resize((new_w, desired_h), Image.Resampling.LANCZOS)

        # padding image
        if desired_w != None and desired_w > new_w:
            new_img = Image.new("L", (desired_w, desired_h), color=0)
            new_img.paste(img, (0, 0))
            img = new_img
        
        return ToTensor()(img) 
    
    def target_loader(self, path):
        label = open(path).read().rstrip('\n')
        return label
    
# ---------------------------------------------------------------
def flist_reader(imgs, labels):
    imlist = []
    for impath in os.listdir(imgs):
        imlabel = os.path.splitext(impath)[0] + '.txt'
        
        impath = imgs + '/' + impath
        imlabel = labels + '/' +  imlabel
        imlist.append((impath, imlabel))
                                    
    return imlist