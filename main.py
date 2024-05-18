import argparse
import torch
import random
import numpy as np
import os
from tqdm import tqdm

from dataset_v2 import DatasetImg_v2
from model.crnn import CRNN
from model.MyCrnn import MyCRNN
from dataset import DatasetImg
from utils.StrLabelConverter import *

from trainer import *

parser = argparse.ArgumentParser()

parser.add_argument('--root', required=True, help='path to root')
parser.add_argument('--alphabet', type=str, default='data/mychar.txt', help='path to char in labels')

parser.add_argument('--imgW', type=int, default=800, help='img width')
parser.add_argument('--dstype', type=str, default='v1', help='version of dataset')
parser.add_argument('--model', type=str, default='MyCRNN', help='type of model')

parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--nepochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--manualSeed', type=int, default=1708, help='reproduce experiemnt')
parser.add_argument('--pretrained', default='', help="path to pretrained model (to continue training)")
parser.add_argument('--savedir', default='checkpoint', help="path to savedir ")

parser.add_argument('--num_hidden', type=int, default=200, help='size of the lstm hidden state')
parser.add_argument('--dropout', type=int, default=0.1, help='dropout')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')

parser.add_argument('--valInterval', type=int, default = 1, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default = 1, help='Interval to be displayed')
opt = parser.parse_args()

random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed) # Comment lại để cho khởi tạo tham số ngẫu nhiên

if __name__ == '__main__':
    device = ( "cuda" if torch.cuda.is_available() else "cpu")
    print("---------------------------------------------------")
    print(f"Using {device} device")
    print("---------------------------------------------------")

    # --------------Tạo Dataset -------------------------------------------------------
    if opt.dstype == 'v1':
        print('Sử dụng dataset_v1')
        train_dataset = DatasetImg(opt.root + '/train/img', opt.root + '/train/label', imgW=opt.imgW)
        test_dataset = DatasetImg(opt.root + '/test/img', opt.root + '/test/label', imgW=opt.imgW)
    elif opt.dstype == 'v2':
        print('Sử dụng dataset_v2')
        train_dataset = DatasetImg_v2(opt.root + '/train/img', opt.root + '/train/label', imgW=opt.imgW)
        test_dataset = DatasetImg_v2(opt.root + '/test/img', opt.root + '/test/label', imgW=opt.imgW)

    with open(os.path.join(opt.alphabet), 'r', encoding='utf-8') as f:
        alphabet = f.read().rstrip()

    converter = StrLabelConverter(alphabet)
    print('Num class: ', converter.numClass)



    # --------------------- Create Model ---------------------------------
    if opt.model == 'MyCRNN':
        model = MyCRNN(converter.numClass, opt.num_hidden, opt.dropout).to(device)
    elif opt.model == 'CRNN':
        model = CRNN(converter.numClass, opt.num_hidden).to(device)

    criterion = torch.nn.CTCLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    trainer = Trainer(model, optimizer, criterion, converter, opt,
                      train_dataset,
                      test_dataset)
    
    if opt.pretrained : trainer.load_pretrained(opt.pretrained)
    
    trainer.train(opt.nepochs)