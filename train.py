import argparse
import torch
import random
import numpy as np
from tqdm import tqdm
import os
from torch.utils.data import random_split

from dataset import DatasetImg

parser = argparse.ArgumentParser()
parser.add_argument('--root', required=True, help='path to root folder')
parser.add_argument('--train', required=True, help='path to dataset')
parser.add_argument('--val', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--imgH', type=int, default=48, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=520, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--gpu', type=int, default=0, help='number of GPUs to use')
parser.add_argument('--pretrained', default='', help="path to pretrained model (to continue training)")
parser.add_argument('--alphabet', type=str, required=True, help='path to char in labels')
parser.add_argument('--expr_dir', required=True, type=str, help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=1, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
parser.add_argument('--valInterval', type=int, default=1, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=1, help='Interval to be displayed')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate for Critic, not used by adadealta')
parser.add_argument('--manualSeed', type=int, default=1234, help='reproduce experiemnt')
opt = parser.parse_args()

random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# --------------Tạo Dataset -------------------------------------------------------
dataset = DatasetImg(opt.imgFolder, opt.labelFolder, opt.imgW, opt.imgH)
train_dataset, test_dataset = random_split(dataset, [8, 2])

train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True)
test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=opt.batch_size,
            shuffle=True)

alphabet = open(os.path.join(opt.root, opt.alphabet)).read().rstrip()
nclass = len(alphabet) + 1
nc = 3

print(len(alphabet), alphabet)
converter = utils.strLabelConverter(alphabet, ignore_case=False)
criterion = CTCLoss()


# --------------------- Create Model ---------------------------------
model = 
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")


    
for epoch in range(1, opt.nepoch+1):
    t = tqdm(iter(train_dataloader), total=len(train_dataloader), desc='Epoch {}'.format(epoch))
    for i, data in enumerate(t):
        for p in crnn.parameters():
            p.requires_grad = True
        crnn.train()

        cost, cer_loss, n = trainBatch(crnn, data, criterion, optimizer)       

        train_loss_avg.add(cost)
        train_cer_avg.add(cer_loss)

    print('[%d/%d] Loss: %f - cer loss: %f' %
            (epoch, opt.nepoch, train_loss_avg.val(), train_cer_avg.val()))
    train_loss_avg.reset()
    train_cer_avg.reset()

    if epoch % opt.valInterval == 0:
        val(crnn, test_loader, criterion)

    # do checkpointing
    if epoch % opt.saveInterval == 0:
        torch.save(
            crnn.state_dict(), '{}/netCRNN_{}.pth'.format(opt.expr_dir, epoch))

