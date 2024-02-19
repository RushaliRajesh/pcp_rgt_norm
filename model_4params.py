import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import pdb

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # self.pre_conv= nn.Conv2d(in_channels=6, out_channels=32, kernel_size=1, stride=1)
        # self.pre_conv= nn.Conv2d(in_channels=6, out_channels=64, kernel_size=1, stride=1)
        self.conv1 = nn.Conv2d(in_channels=33, out_channels=32, kernel_size=3, stride=1, bias=False)
        self.conv2 = nn.Conv2d(32, 16, 3, 1, bias=False)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(1920, 64)
        # self.fc1 = nn.Linear(720, 64)
        self.fc2 = nn.Linear(64, 32, bias=False)
        self.fc3 = nn.Linear(32, 3, bias=False)
        # self.act = nn.LeakyReLU()
        self.act = nn.ReLU()
        self.bn1 = nn.InstanceNorm2d(64)
        self.bn2 = nn.InstanceNorm2d(64)
        self.bn3 = nn.InstanceNorm2d(64)
        self.bn4 = nn.InstanceNorm2d(32)
        self.bn5 = nn.InstanceNorm2d(16)
        self.drop1 = nn.Dropout(p=0.2)
        self.drop2 = nn.Dropout(p=0.2)
      

    def forward(self, patches, init):

        x = self.act((self.conv1(patches)))
        # print(x.shape)
        x = self.act((self.conv2(x)))
        # print(x.shape)
        x = self.flat(x)
        # print(x.shape)
        x_in = self.act(self.init_fc1(init))
        # print(x_in.shape)
        x_in = self.act(self.init_fc2(x_in))
        # print(x_in.shape)
        x = torch.cat((x, x_in), dim=-1)
        # print(x.shape)
        x = self.act(self.drop1(self.fc1(x)))
        # print(x.shape)
        x = self.act(self.drop2(self.fc2(x)))
        # print(x.shape)
        x = self.fc3(x)
        # print(x.shape)
        x = torch.tanh(x)
        # print(x.shape)
        
        return x
    
class CNN_nopos(nn.Module):
    def __init__(self):
        super(CNN_nopos, self).__init__()
        # self.pre_conv= nn.Conv2d(in_channels=6, out_channels=32, kernel_size=1, stride=1)
        # self.pre_conv= nn.Conv2d(in_channels=6, out_channels=64, kernel_size=1, stride=1)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 16, 3, 1)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(2880, 64)
        # self.fc1 = nn.Linear(720, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 4)

        self.init_fc1 = nn.Linear(3, 32)
        self.init_fc2 = nn.Linear(32, 64)

        self.act = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.2)
        self.drop2 = nn.Dropout(p=0.2)
      

    def forward(self, patches, init):

        x = self.act((self.conv1(patches)))
        # print(x.shape)
        x = self.act((self.conv2(x)))
        # print(x.shape)
        x = self.flat(x)
        # print(x.shape)
        x_in = self.act(self.init_fc1(init))
        # print(x_in.shape)
        x_in = self.act(self.init_fc2(x_in))
        # print(x_in.shape)
        x = torch.cat((x, x_in), dim=-1)
        # print(x.shape)
        x = self.act(self.drop1(self.fc1(x)))
        # print(x.shape)
        x = self.act(self.drop2(self.fc2(x)))
        # print(x.shape)
        x = self.fc3(x)
        # print(x.shape)
        x = torch.tanh(x)
        # print(x.shape)
        
        return x
    
    
class CNN_nopos_udp(nn.Module):
    def __init__(self):
        super(CNN_nopos, self).__init__()
        # self.pre_conv= nn.Conv2d(in_channels=6, out_channels=32, kernel_size=1, stride=1)
        # self.pre_conv= nn.Conv2d(in_channels=6, out_channels=64, kernel_size=1, stride=1)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(11328, 64)
        # self.fc1 = nn.Linear(720, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 4)

        self.init_fc1 = nn.Linear(3, 32)
        self.init_fc2 = nn.Linear(32, 64)

        self.act = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.2)
        self.drop2 = nn.Dropout(p=0.2)
      

    def forward(self, patches, init):

        x = self.act((self.conv1(patches)))
        # print(x.shape)
        x = self.act((self.conv2(x)))
        # print(x.shape)
        x = self.flat(x)
        # print(x.shape)
        x_in = self.act(self.init_fc1(init))
        # print(x_in.shape)
        x_in = self.act(self.init_fc2(x_in))
        # print(x_in.shape)
        x = torch.cat((x, x_in), dim=-1)
        # print(x.shape)
        x = self.act(self.drop1(self.fc1(x)))
        # print(x.shape)
        x = self.act(self.drop2(self.fc2(x)))
        # print(x.shape)
        x = self.fc3(x)
        # print(x.shape)
        x = torch.tanh(x)
        # print(x.shape)
        
        return x
    
class CNN_pos(nn.Module):

    def __init__(self):
        super(CNN_pos, self).__init__()
        # self.pre_conv= nn.Conv2d(in_channels=6, out_channels=32, kernel_size=1, stride=1)
        # self.pre_conv= nn.Conv2d(in_channels=6, out_channels=64, kernel_size=1, stride=1)
        self.conv1 = nn.Conv2d(in_channels=15, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(11296, 64)
        # self.fc1 = nn.Linear(720, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 4)

        self.init_fc1 = nn.Linear(3, 16)
        self.init_fc2 = nn.Linear(16, 32)

        self.act = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.1)
      

    def forward(self, patches, init):

        x = self.act((self.conv1(patches)))
        # print(x.shape)
        x = self.act((self.conv2(x)))
        # print(x.shape)
        x = self.flat(x)
        # print(x.shape)
        x_in = self.act(self.init_fc1(init))
        # print(x_in.shape)
        x_in = self.act(self.init_fc2(x_in))
        # print(x_in.shape)
        x = torch.cat((x, x_in), dim=-1)
        # print(x.shape)
        x = self.act(self.fc1(x))
        # print(x.shape)
        x = self.act(self.drop2(self.fc2(x)))
        # print(x.shape)
        x = self.fc3(x)
        # print(x.shape)
        x = torch.tanh(x)
        # print(x.shape)
        
        return x
    

class mlp (nn.Module):
    def __init__(self):
        super(mlp,self).__init__()
        self.fc1 = nn.Linear(3*12*26, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(64, 4)
        self.act = nn.ReLU()

        self.fc5 = nn.Linear(3, 32)
        # self.fc6 = nn.Linear()

    def forward(self, patches, init):
        x = self.act(self.fc1(patches))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))

        y = self.act(self.fc5(init))
        x = torch.cat((x, y), dim=-1)

        x = torch.tanh(self.fc4(x))

        return x



if __name__ == '__main__':
    dumm = torch.rand(4,15,12,26)
    dumm_norm = torch.rand(4,3)
    model = CNN_pos()
    out = model(dumm, dumm_norm)
    # model = mlp()
    # out = model(dumm, dumm_norm)
    print(out.shape)

