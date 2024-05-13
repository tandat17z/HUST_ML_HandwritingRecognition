
import torch
import torch.nn as nn
from model.modules import *

class CRNN(nn.Module):

    def __init__(self, num_class, hidden_size, input_channel = 1, output_channel = 512):
        super(CRNN, self).__init__()
        print('>>>> use CRNN-------------\n')

        """ FeatureExtraction """
        self.FeatureExtraction = VGG_FeatureExtractor(input_channel, output_channel)
        # self.FeatureExtraction_output = output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(output_channel, hidden_size, hidden_size),
                BidirectionalLSTM(hidden_size, hidden_size, hidden_size))
        self.SequenceModeling_output = hidden_size
        
        """ Prediction """
        self.Prediction = nn.Linear(self.SequenceModeling_output, num_class)
        
    def forward(self, input):
        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        contextual_feature = self.SequenceModeling(visual_feature)
        
        """ Prediction stage """
        prediction = self.Prediction(contextual_feature.contiguous())
        
        return prediction
