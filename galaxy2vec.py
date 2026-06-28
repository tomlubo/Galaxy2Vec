'''
parts of this code were inspired by https://github.com/sthalles/SimCLR?utm_source=catalyzex.com
and the paper
Ting Chen et al. A Simple Framework for Contrastive Learning of Visual Representations. 2020. arXiv:
2002.05709 [cs.LG]. url: https://arxiv.org/abs/2002.05709.

'''



import torch

import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size = 3, pool = True):
        super().__init__()

        padding = kernel_size // 2

        layers = [nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, padding = padding, bias = False),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(inplace = True)]
        
        if pool:
            layers.append(nn.MaxPool2d(kernel_size = 2, stride = 2))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
        
    
class GalaxyEncoder(nn.Module):

    def __init__(self, in_channels = 3, feature_dim = 128):

        super().__init__()

        self.features = nn.Sequential(ConvBlock(in_channels, 32, pool = True),
                                      ConvBlock(32, 64, pool = True),
                                      ConvBlock(64, 128, pool = True),
                                      ConvBlock(128, 256, pool = True),
                                      ConvBlock(256, 256, pool = True) )
        

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, feature_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
    

class RegressionHead(nn.Module):

    def __init__(self, in_dim = 256, hidden_dim = 128, dropout = 0.2):

        super().__init__()

        self.regressor = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace = True),nn.Dropout(dropout),
                                       nn.Linear(hidden_dim, 1), nn.Sigmoid())
        
    def forward(self, x):
        return self.regressor(x)
    

class ProjectionHead(nn.Module):

    def __init__(self, in_dim = 256, hidden_dim = 256, proj_dim = 128, dropout = 0.2):

        super().__init__()

        self.head = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace = True), nn.Dropout(dropout),
                                  nn.Linear(hidden_dim, proj_dim))
        
    def forward(self,x):
        return F.normalize(self.head(x), dim = -1)
    


    
class Galaxy2VecRegressor(nn.Module):

    def __init__(self, in_channels, feature_dim = 256, hidden_dim = 128, dropout = 0.2):

        super().__init__()

        self.encoder = GalaxyEncoder(in_channels, feature_dim)

        self.regressor = RegressionHead(feature_dim, hidden_dim, dropout)

    def forward(self,x):

        x = self.encoder(x)
        return self.regressor(x)
        

class Galaxy2VecContrastive(nn.Module):

    def __init__(self, in_channels, feature_dim =256, proj_dim = 128):

        super().__init__()

        self.encoder = GalaxyEncoder(in_channels, feature_dim)

        self.proj_head = ProjectionHead(feature_dim, feature_dim, proj_dim, 0.2)

    def forward(self, x, return_features = False):
        emb = self.encoder(x)
        x = self.proj_head(emb)

        if return_features:
            return x, emb
        return x
