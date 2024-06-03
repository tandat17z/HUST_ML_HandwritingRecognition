import argparse
import torch
import random
import numpy as np
import os
from tqdm import tqdm

from dataset import DatasetImg
from model.crnn import CRNN

from dataset import DatasetImg
from tools.StrLabelConverter import *

from trainer import *

parser = argparse.ArgumentParser()

parser.add_argument('--dstrain', required=True, help='path to train')
parser.add_argument('--dsval', required=True, help='path to val')
parser.add_argument('--alphabet', type=str, default='data/alphabet.txt', help='path to char in labels')

parser.add_argument('--imgW', type=int, default=768, help='img width')
parser.add_argument('--threshold', type=int, default=30, help='threshold')
parser.add_argument('--num_hidden', type=int, default=128, help='size of the lstm hidden state')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

parser.add_argument('--savedir', type=str, required = True, help="path to savedir ")

parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
parser.add_argument('--nepochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--manualSeed', type=int, default=1708, help='reproduce experiemnt')
parser.add_argument('--pretrained', default='', help="path to pretrained model (to continue training)")

parser.add_argument('--valInterval', type=int, default = 1, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default = 1, help='Interval to be saved')
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
    
    train_dataset = DatasetImg(opt.dstrain + '/img', opt.dstrain + '/label', imgW=opt.imgW, threshold=opt.threshold)
    val_dataset = DatasetImg(opt.dsval + '/img', opt.dsval + '/label', imgW=opt.imgW, threshold=opt.threshold)

    with open(os.path.join(opt.alphabet), 'r', encoding='utf-8') as f:
        alphabet = f.read().rstrip()

    converter = StrLabelConverter(alphabet)
    print('Num class: ', converter.numClass)

    # --------------------- Create Model ---------------------------------
    model = CRNN(converter.numClass, opt.num_hidden, opt.dropout).to(device)
    criterion = torch.nn.CTCLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    trainer = Trainer(model, optimizer, criterion, converter, opt,
                      train_dataset,
                      val_dataset)
    
    if opt.pretrained : trainer.load_pretrained(opt.pretrained)
    
    trainer.train(opt.nepochs)