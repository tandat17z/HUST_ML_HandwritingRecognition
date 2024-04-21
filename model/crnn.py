import sys
import torch.nn as nn
import torch

class CRNN(nn.Module):

    def __init__(self, nclass, num_hidden, dropout = 0):
        super(CRNN, self).__init__()

        ks = [ 3,      3,      3,      3,      3,   3, 1]
        ss = [ 1,      1, (2, 1),      1, (2, 1),   1, 1]
        ps = [ 1,      1,      1,      1,      1,   1, 1]
        nm = [64,    128,    128,    256,    256, 512, 512]

        cnn = nn.Sequential()
        def convRelu(i):
            nIn = 1 if i == 0 else nm[i - 1] 
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            # cnn.add_module('conv{0}'.format(i), nn.Conv2d(nIn, nOut, 3, 1, 1))
            cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        # input : (C, H, W) - (1, 32, 512)
        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d((2, 2)))  # 64, 16, 256
        convRelu(1) 
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d((2, 2)))  # 128, 4, 128
        convRelu(2) 
        convRelu(3) 
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d((2, 2)))  # 256, 1, 64
        convRelu(4) 
        convRelu(5)

        self.cnn = cnn
        self.linear1 = nn.Linear(64, 128, bias = True)
        self.dropout1 = nn.Dropout(dropout)

        # BiLSTM
        self.biLSTM = nn.LSTM(512, num_hidden, bidirectional=True, batch_first = True)
        self.dropout2 = nn.Dropout(dropout)

        self.linear2 = nn.Linear(num_hidden * 2, nclass, bias = True)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, input):
        # conv features
        x1 = self.cnn(input)

        x2 = self.linear1(x1)
        x2 = self.dropout1(x2)

        x2 = torch.squeeze(x2, 2)
        x2 = x2.permute(0, 2, 1)

        x3, _  = self.biLSTM(x2)
        x3 = self.dropout2(x3)
        out = self.linear2(x3)
        out = self.dropout3(out)

        return out
