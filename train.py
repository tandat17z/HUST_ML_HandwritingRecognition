import argparse
import torch
import random
import numpy as np
import os

from model.crnn import CRNN
from model.MyCrnn import MyCRNN
from dataset import DatasetImg
from utils.StrLabelConverter import *

from trainer import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='path to data folder')
    parser.add_argument('--alphabet', type=str, default='data/mychar.txt', help='path to char in labels')
    
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--manualSeed', type=int, default=1708, help='reproduce experiemnt')
    parser.add_argument('--pretrained', default='', help="path to pretrained model (to continue training)")

    parser.add_argument('--num_hidden', type=int, default=125, help='size of the lstm hidden state')
    parser.add_argument('--dropout', type=int, default=0.1, help='dropout')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for Critic, not used by adadealta')

    parser.add_argument('--valInterval', type=int, default = 1, help='Interval to be displayed')
    parser.add_argument('--saveInterval', type=int, default = 1, help='Interval to be displayed')
    opt = parser.parse_args()

    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed) # Comment lại để cho khởi tạo tham số ngẫu nhiên

    device = ( "cuda" if torch.cuda.is_available() else "cpu")
    print("---------------------------------------------------")
    print(f"Using {device} device")
    print("---------------------------------------------------")

    # --------------Tạo Dataset -------------------------------------------------------
    dataset = DatasetImg(opt.data + '/img', opt.data + '/label')

    with open(os.path.join(opt.alphabet), 'r', encoding='utf-8') as f:
        alphabet = f.read().rstrip()
    converter = StrLabelConverter(alphabet)
    print('Num class: ', converter.numClass)

    # --------------------- Create Model ---------------------------------
    model = MyCRNN(converter.numClass, opt.num_hidden, opt.dropout).to(device)
    # print(f"Model structure: {model}\n\n")
    # for name, param in model.named_parameters():
    #     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

    criterion = torch.nn.CTCLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    epoch = 0
    if opt.pretrained:
        checkpoint_path = opt.pretrained
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']

    trainer = Trainer(model, 
                      optimizer = optimizer,
                      criterion = criterion, 
                      converter = converter,
                      dataset = dataset,
                      batch_size = opt.batch_size)
    trainer.train(opt.nepoch, opt.valInterval, opt.saveInterval, start_epoch = epoch)