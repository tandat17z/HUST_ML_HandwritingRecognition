import torch
from utils import utils
from PIL import Image
from torchvision.transforms import ToTensor

def img_loader(imgpath, imgH, imgW):
    img = Image.open(imgpath)

    # Resize hình ảnh + thêm padding (nếu cần)
    desired_w, desired_h = imgW, imgH #(width, height)
    img_w, img_h = img.size  # old_size[0] is in (width, height) format
    ratio = 1.0*img_w/img_h
    new_w = int(desired_h*ratio)
    new_w = min(desired_w, new_w)
    img = img.resize((new_w, desired_h), Image.Resampling.LANCZOS)

    # padding image
    if desired_w != None and desired_w > new_w:
        new_img = Image.new("RGB", (desired_w, desired_h), color=(0, 0, 0))
        new_img.paste(img, (0, 0))
        img = new_img
    # img = np.array(img)
    # return torch.from_numpy(img)
    # img = img.resize((self.imgW, self.imgH), Image.Resampling.LANCZOS)
    return ToTensor()(img)

class DatasetImg_v3(torch.utils.data.Dataset):
    def __init__(self, imgFolder, labelFolder, imgH = 32, imgW = 800):
        self.imgH = imgH
        self.imgW = imgW
        self.imlist = utils.flist_reader(imgFolder, labelFolder)

    def __getitem__(self, index):
        idx = index 
        imgpath, imglabel = self.imlist[idx]
        img = img_loader(imgpath, self.imgH, self.imgW)
        
        target = utils.target_loader(imglabel)
        return img, target


    def __len__(self):
        return len(self.imlist)
    
