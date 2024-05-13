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

# from trainer import *

parser = argparse.ArgumentParser()

parser.add_argument('--train', required=True, help='path to traindata folder')
parser.add_argument('--test', required=True, help='path to traindata folder')
parser.add_argument('--alphabet', type=str, default='data/mychar.txt', help='path to char in labels')

parser.add_argument('--imgW', type=int, default=512, help='img width')
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

parser.add_argument('--valInterval', type=int, default = 5, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default = 5, help='Interval to be displayed')
opt = parser.parse_args()

random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed) # Comment lại để cho khởi tạo tham số ngẫu nhiên

def GetInputCTCLoss(preds, labels):
    b, l, c = preds.shape
    preds_ = preds.permute(1, 0, 2).to('cpu')
    preds_lengths = torch.full(size=(b,), fill_value=l, dtype=torch.long).to('cpu')

    targets, target_lengths = converter.encode(labels)
    targets = targets.to('cpu')
    target_lengths = target_lengths.to('cpu')

    return preds_, preds_lengths, targets, target_lengths

if __name__ == '__main__':
    device = ( "cuda" if torch.cuda.is_available() else "cpu")
    print("---------------------------------------------------")
    print(f"Using {device} device")
    print("---------------------------------------------------")

    # --------------Tạo Dataset -------------------------------------------------------

    if opt.dstype == 'v1':
        print('Sử dụng dataset_v1')
        train_dataset = DatasetImg(opt.train + '/img', opt.train + '/label', imgW=opt.imgW)
        test_dataset = DatasetImg(opt.test + '/img', opt.test + '/label', imgW=opt.imgW)
    elif opt.dstype == 'v2':
        print('Sử dụng dataset_v2')
        train_dataset = DatasetImg_v2(opt.train + '/img', opt.train + '/label')
        test_dataset = DatasetImg_v2(opt.test + '/img', opt.test + '/label')

    train_dataloader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=opt.batch_size,
                    shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=opt.batch_size,
                    shuffle=True)
    
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

    start_epoch = 0
    log = list()
    if opt.pretrained:
        checkpoint_path = opt.pretrained
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        log = checkpoint['log']

    # Training----------------------------------------------------------------------------------
    for epoch in range(start_epoch + 1, start_epoch + opt.nepochs + 1):
        print('Epoch: ', epoch)
        # Train -------------------------
        model.train(True)
        total_loss = 0
        levenshtein_loss = 0

        t = tqdm(iter(train_dataloader), total=len(train_dataloader))
        for batch_idx, (imgs, labels) in enumerate(t):
            imgs = imgs.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            
            preds_, preds_lengths, targets, target_lengths = GetInputCTCLoss(preds, labels)
            loss = criterion(preds_.log_softmax(2), targets, preds_lengths, target_lengths) # ctc_loss chỉ dùng với cpu, dùng với gpu phức tạp hơn thì phải
            assert (not torch.isnan(loss) and not torch.isinf(loss)), "Loss value is NaN or Inf"
            
            # Backward -------------------------------------------------
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().item()

            _, enc_preds = preds.max(2)
            sim_preds = converter.decode(enc_preds.view(-1), preds_lengths, raw = False)
            levenshtein_loss += converter.Levenshtein_loss(sim_preds, labels)

        total_loss = total_loss/train_dataloader.sampler.num_samples *opt.batch_size
        levenshtein_loss = levenshtein_loss/train_dataloader.sampler.num_samples 
        print('Epoch: [{}/{}]\t avg_Loss/batch = {:.4f} \t Levenshtein_Loss/sentence = {:.2f}'.format(epoch, start_epoch + opt.nepochs, total_loss, levenshtein_loss))
        log.append({
            'type': 'train',
            'epoch': epoch,
            'avg_Loss': total_loss,
            'levenshtein_Loss': levenshtein_loss
        })

        # Val ---------------------------
        if epoch % opt.valInterval == 0: 
            print("Tester.eval...")
            model.eval()
            with torch.no_grad():
                total_loss = 0
                levenshtein_loss = 0

                t = tqdm(iter(test_dataloader), total=len(test_dataloader))
                for batch_idx, (imgs, labels) in enumerate(t):
                    imgs = imgs.to(device)
                    preds = model(imgs)
                    
                    preds_, preds_lengths, targets, target_lengths = GetInputCTCLoss(preds, labels)
                    loss = criterion(preds_.log_softmax(2), targets, preds_lengths, target_lengths) # ctc_loss chỉ dùng với cpu, dùng với gpu phức tạp hơn thì phải
                    total_loss += loss.detach().item()

                    _, enc_preds = preds.max(2)
                    sim_preds = converter.decode(enc_preds.view(-1), preds_lengths, raw = False)
                    levenshtein_loss += converter.Levenshtein_loss(sim_preds, labels)
            # Display ----------------------------------------  
            if True:
                raw_preds = converter.decode(enc_preds.view(-1), preds_lengths, raw = True)
                i = 5
                for raw_pred, pred, gt in zip(raw_preds, sim_preds, labels):
                    print('='*30)
                    print(f'raw: {raw_pred}')
                    print(f'pred_text: {pred}', )
                    print(f'gt: {gt}')
                    i -= 1
                    if( i == 0): break

            total_loss = total_loss/test_dataloader.sampler.num_samples *opt.batch_size
            levenshtein_loss = levenshtein_loss/test_dataloader.sampler.num_samples
            log.append({
                'type': 'val',
                'epoch': epoch,
                'avg_Loss': total_loss,
                'levenshtein_Loss': levenshtein_loss
            })

            print('--> Val: \t avg_Loss/batch = {:.4f} \t Levenshtein_Loss/sentence = {:.2f}'.format(total_loss, levenshtein_loss))
        
        # Save --------------------------
        if epoch % opt.saveInterval == 0:
            print('Saving Model...\n')

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),  # Lưu trạng thái của mô hình
                'optimizer_state_dict': optimizer.state_dict(),  # Lưu trạng thái của optimizer
                'log': log
            }
            torch.save(checkpoint, opt.savedir +'/' + opt.model +  f'/checkpoint-{epoch}.pth.tar')