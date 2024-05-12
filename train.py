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

parser = argparse.ArgumentParser()
parser.add_argument('--train', required=True, help='path to traindata folder')
parser.add_argument('--test', required=True, help='path to traindata folder')
parser.add_argument('--alphabet', type=str, default='data/mychar.txt', help='path to char in labels')

parser.add_argument('--imgW', type=int, default=512, help='img width')

parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--nepochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--manualSeed', type=int, default=1708, help='reproduce experiemnt')
parser.add_argument('--pretrained', default='', help="path to pretrained model (to continue training)")

parser.add_argument('--num_hidden', type=int, default=200, help='size of the lstm hidden state')
parser.add_argument('--dropout', type=int, default=0.1, help='dropout')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate for Critic, not used by adadealta')

parser.add_argument('--valInterval', type=int, default = 1, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default = 5, help='Interval to be displayed')
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
    train_dataset = DatasetImg(opt.train + '/img', opt.train + '/label', imgW=opt.imgW)
    test_dataset = DatasetImg(opt.test + '/img', opt.test + '/label', imgW=opt.imgW)

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

    start_epoch = 0
    if opt.pretrained:
        checkpoint_path = opt.pretrained
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']

    trainer = Trainer(model, 
                      optimizer = optimizer,
                      criterion = criterion, 
                      converter = converter,
                      train_dataset = train_dataset,
                      batch_size = opt.batch_size)
    
    tester = Tester(model,
                      criterion = criterion, 
                      converter = converter,
                      test_dataset = test_dataset,
                      batch_size = opt.batch_size)
    
    for epoch in range(start_epoch + 1, start_epoch + opt.nepochs + 1):
        print('Epoch: ', epoch)
        # Train -------------------------
        total_loss, levenshtein_loss = trainer.train()
        print('Epoch: [{}/{}]\t avg_Loss = {:.4f} \t Levenshtein Loss per 1 sentence = {:.2f}'.format(epoch, start_epoch + opt.nepochs, total_loss, levenshtein_loss))
        
        # Val ---------------------------
        if epoch % opt.valInterval == 0: 
            total_loss, levenshtein_loss = tester.eval()
            print('--> Val: \t avg_Loss = {:.4f} \t Levenshtein Loss per 1 sentence = {:.2f}'.format(total_loss, levenshtein_loss))
        
        # Save --------------------------
        if epoch % opt.saveInterval == 0:
            print('Saving Model...\n')

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),  # Lưu trạng thái của mô hình
                'optimizer_state_dict': optimizer.state_dict(),  # Lưu trạng thái của optimizer
            }
            torch.save(checkpoint, f'pretrain/model_{epoch}.pth.tar')