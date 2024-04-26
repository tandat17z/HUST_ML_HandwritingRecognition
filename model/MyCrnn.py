
import torch.nn as nn
import torch

class MyCRNN(nn.Module):

    def __init__(self, nclass, num_hidden, dropout = 0):
        print('>>>> use MyCrnn-------------\n')
        super(MyCRNN, self).__init__()

        # ks = [ 3,   3,   3,   3,   3,   3,   1,   1,   1]
        # ss = [ 1,   1,   1,   1,   1,   1,   1,   1,   1]
        # ps = [ 1,   1,   1,   1,   1,   1,   1,   1,   1]
        nm = [64, 128, 256, 256, 256, 512, 512, 512, 512]

        cnn = nn.Sequential()
        def convRelu(i):
            nIn = 1 if i == 0 else nm[i - 1] 
            nOut = nm[i]
            # cnn.add_module('conv{0}'.format(i),
            #                nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            cnn.add_module('conv{0}'.format(i), nn.Conv2d(nIn, nOut, 3, 1, 1))
            cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        # input : (C, H, W) - (1, 32, 512)
        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d((2, 2)))  # 64, 16, 256
        convRelu(1) 
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d((2, 2)))  # 128, 8, 128
        convRelu(2) 
        convRelu(3) 
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d((2, 1)))  # 256, 4, 128
        convRelu(4) 
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3), nn.MaxPool2d((2, 1)))  # 512, 2, 128
        convRelu(6) 
        convRelu(7)
        cnn.add_module('pooling{0}'.format(4), nn.MaxPool2d((2, 1)))  # 512, 1, 128

        self.cnn = cnn

        # BiLSTM
        self.biLSTM1 = nn.LSTM(512, num_hidden, bidirectional=True, batch_first = True)
        self.dropout1 = nn.Dropout(dropout)

        self.linear = nn.Linear(num_hidden * 2, nclass, bias = True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input):
        # conv features
        x1 = self.cnn(input) # b, 512, 1, 128
        x1 = torch.squeeze(x1, 2) # b, 512, 128
        x1 = x1.permute(0, 2, 1)  # b, 128, 512

        x2, _  = self.biLSTM1(x1) # b, 128, num_hidden*2
        x2 = self.dropout1(x2) 

        x3 = self.linear(x2) # b, 128, num_class
        out = self.dropout(x3)

        return out
