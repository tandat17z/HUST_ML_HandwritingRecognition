from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor
import os
import torch


def GetInputCTCLoss(converter, preds, labels):
    b, l, c = preds.shape
    preds_ = preds.permute(1, 0, 2).to('cpu')
    preds_lengths = torch.full(size=(b,), fill_value=l, dtype=torch.long).to('cpu')

    targets, target_lengths = converter.encode(labels)
    targets = targets.to('cpu')
    target_lengths = target_lengths.to('cpu')

    return preds_, preds_lengths, targets, target_lengths


def flist_reader(imgs, labels):
    imlist = []
    for impath in os.listdir(imgs):
        imlabel = os.path.splitext(impath)[0] + '.txt'
        
        impath = imgs + '/' + impath
        imlabel = labels + '/' +  imlabel
        imlist.append((impath, imlabel))
                                    
    return imlist

def img_loader(path, imgH = 32, imgW = 512, scale = False,  alignment = 'left', threshold = 40):
    img = Image.open(path).convert('L')
    img = img.point(lambda p: 255 - p if 255 - p >= threshold else 0) # chuyển background về màu đen 0
    
    img = cropImg(img)

    # Resize hình ảnh + thêm padding (nếu cần)
    desired_w, desired_h = imgW, imgH #(width, height)
    img_w, img_h = img.size  # old_size[0] is in (width, height) format
    ratio = 1.0*img_w/img_h
    new_w = int(desired_h*ratio)
    if scale:
        new_w = int(new_w * scale)
    new_w = min(desired_w, new_w)
    img = img.resize((new_w, desired_h), Image.Resampling.LANCZOS)

    # padding image
    if desired_w != None and desired_w > new_w:
        new_img = Image.new("L", (desired_w, desired_h), color=0)
        
        if alignment == 'left':
            new_img.paste(img, (0, 0))
        elif alignment == 'center':
            new_img.paste(img, (int((desired_w - new_w) / 2), 0))
        elif alignment == 'right':
            new_img.paste(img, (int((desired_w - new_w)), 0))
        img = new_img
    
    # img = np.array(img)
    # return torch.from_numpy(img)
    # img = img.resize((self.imgW, self.imgH), Image.Resampling.LANCZOS)
    return ToTensor()(img)
    
def target_loader(path):
    with open(path, 'r', encoding='utf-8') as f:
        label = f.read().rstrip('\n')
    return label.strip()

def cropImg(img):
    # Cắt bỏ khoảng trống bị thừa xung quanh
    img_array = np.array(img)

    non_empty_columns = np.where(img_array.max(axis=0) > 0)[0]
    non_empty_rows = np.where(img_array.max(axis=1) > 0)[0]
    cropped_img = img_array[min(non_empty_rows):max(non_empty_rows) + 1,
                        min(non_empty_columns):max(non_empty_columns) + 1]
    cropped_img = Image.fromarray(cropped_img)
    return cropped_img
