import argparse
import torch
import random
import numpy as np
from tqdm import tqdm
import os
from torch.utils.data import random_split

from model.crnn import CRNN
from dataset import DatasetImg
from utils.utils import *
from trainer import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='path to data folder')
    parser.add_argument('--alphabet', type=str, required=True, help='path to char in labels')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
    parser.add_argument('--imgW', type=int, default=512, help='the width of the input image to network')

    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--nepoch', type=int, default=2, help='number of epochs to train for')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--manualSeed', type=int, default=1234, help='reproduce experiemnt')
    parser.add_argument('--pretrained', default='', help="path to pretrained model (to continue training)")

    parser.add_argument('--num_hidden', type=int, default=100, help='size of the lstm hidden state')
    parser.add_argument('--dropout', type=int, default=0.1, help='dropout')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for Critic, not used by adadealta')

    parser.add_argument('--valInterval', type=int, default=5, help='Interval to be displayed')
    parser.add_argument('--saveInterval', type=int, default=5, help='Interval to be displayed')
    # parser.add_argument('--', type=int, default=1, help='Interval to be displayed')

    # parser.add_argument('--train', required=True, help='path to dataset')
    # parser.add_argument('--val', required=True, help='path to dataset')
    # parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    # parser.add_argument('--gpu', type=int, default=0, help='number of GPUs to use')
    # parser.add_argument('--expr_dir', required=True, type=str, help='Where to store samples and models')
    # parser.add_argument('--displayInterval', type=int, default=1, help='Interval to be displayed')
    # parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
    opt = parser.parse_args()

    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    # torch.manual_seed(opt.manualSeed) # Comment lại để cho khởi tạo tham số ngẫu nhiên

    # if torch.cuda.is_available() :
    #     print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    device = ( "cuda" if torch.cuda.is_available() else "cpu")
    print("---------------------------------------------------")
    print(f"Using {device} device")
    print("---------------------------------------------------")

    # --------------Tạo Dataset -------------------------------------------------------
    dataset = DatasetImg(opt.data + '/img', opt.data + '/label', opt.imgW, opt.imgH)

    with open(os.path.join(opt.alphabet), 'r', encoding='utf-8') as f:
        alphabet = f.read().rstrip()
    # print(alphabet)
    converter = strLabelConverter(alphabet, ignore_case=True)

    # --------------------- Create Model ---------------------------------
    model = CRNN(converter.numClass, opt.num_hidden, opt.dropout).to(device)
    # print(f"Model structure: {model}\n\n")
    # for name, param in model.named_parameters():
    #     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

    criterion = torch.nn.CTCLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    epoch = 0
    if opt.pretrained:
        checkpoint_path = opt.pretrain
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']

    trainer = Trainer(opt, model, 
                      criterion = criterion, 
                      optimizer = optimizer,
                      dataset = dataset,
                      converter = converter)
    trainer.train(opt.nepoch, opt.valInterval, opt.saveInterval, start_epoch = epoch)