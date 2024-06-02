
import torch.nn as nn
import torch

class CRNN(nn.Module):
    def __init__(self, nclass, num_hidden, dropout = 0.1):
        print('>>>> use Crnn-------------\n')
        super(CRNN, self).__init__()

        ks = [ 3,  3,   3,   3,   3,   3,  2]
        ss = [ 1,  1,   1,   1,   1,   1,  1]
        ps = [ 1,  1,   1,   1,   1,   1,  0]
        nm = [32, 64, 128, 128, 256, 256, 256]

        cnn = nn.Sequential()
        def convRelu(i):
            nIn = 1 if i == 0 else nm[i - 1] 
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        # input : (C, H, W) - (1, 32, 768)
        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d((2, 2)))  # 32, 16, 384
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d((2, 2)))  # 64, 8, 192
        convRelu(2) 
        convRelu(3) 
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d((2, 1)))  # 128, 4, 192
        convRelu(4)
        convRelu(5) 
        cnn.add_module('pooling{0}'.format(3), nn.MaxPool2d((2, 1)))  # 256, 2, 192
        convRelu(6) # 256, 1, 191
        self.cnn = cnn
        self.dropout_cnn = nn.Dropout(dropout)
    
        # BiLSTM
        self.biLSTM1 = nn.LSTM(nm[-1], num_hidden, bidirectional=True, batch_first = True)
        self.dropout1 = nn.Dropout(dropout)

        self.biLSTM2 = nn.LSTM(num_hidden*2, num_hidden, bidirectional=True, batch_first = True)
        self.dropout2 = nn.Dropout(dropout)

        # Linear
        self.linear = nn.Linear(num_hidden * 2, nclass)
        
    def forward(self, input):
        # conv features
        x1 = self.cnn(input) # b, 256, 1, 127
        x1 = self.dropout_cnn(x1)
        x1 = torch.squeeze(x1, 2) # b, 256, 127
        x1 = x1.permute(0, 2, 1)  # b, 127, 256

        x2, _  = self.biLSTM1(x1) # b, 127, num_hidden*2
        x2 = self.dropout1(x2) 

        x3, _ = self.biLSTM2(x2) # b, 127, num_hidden*2
        x3 = self.dropout2(x3)

        out = self.linear(x2) # b, 127, num_class

        return out
