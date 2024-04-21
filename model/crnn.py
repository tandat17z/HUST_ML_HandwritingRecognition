import sys
import torch.nn as nn
import torch

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut, dropout=0):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True, batch_first = False)
        self.embedding = nn.Linear(nHidden * 2, nOut)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        recurrent = self.dropout(recurrent)
        N, L, h = recurrent.size()
        t_rec = recurrent.view(N * L, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(N, L, -1)

        return output


class CRNN(nn.Module):

    def __init__(self, nclass, nh):
        super(CRNN, self).__init__()

        ks = [3, 3, 3, 3, 3, 3, 2]
        ss = [1, 1, 1, 1, 1, 1, 1]
        ps = [1, 1, 1, 1, 1, 1, 0]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()
        # input : (C, H, W) - (1, 32, 512)
        def convRelu(i):
            nIn = 1 if i == 0 else nm[i - 1] 
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0) # -> (64, 32, 512)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64, 16, 256
        convRelu(1) # -> (128, 16, 256)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128, 8, 128
        convRelu(2) # -> (256, 8, 128)
        convRelu(3) # -> (256, 8, 128)
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d((2, 2)))  # 256x4x64
        convRelu(4) # -> (512, 4, 64)
        convRelu(5) # -> (512, 4, 64)
        convRelu(6) # ->(512, 4, 64)
        cnn.add_module('pooling{0}'.format(3), nn.MaxPool2d((2, 2), (1, 2), (0, 0)))  # 512, 4, 32

        self.cnn = cnn

        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh, 0),
            BidirectionalLSTM(nh, nh, nclass, 0)
            )

    def forward(self, input):
        # conv features
        conv = self.cnn(input)

        b, c, h, w = conv.size()
        # assert h == 1, "the height of conv must be 1"
        # conv = conv.squeeze(2)
        conv = conv.view(b, c, -1)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        return output
