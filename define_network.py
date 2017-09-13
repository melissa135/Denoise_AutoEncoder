import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class AutoEncoder_1(nn.Module):
    def __init__(self, input_size=48, layer1_size=20):
        super(AutoEncoder_1, self).__init__()

	self.input_size = input_size       
	self.layer1_size = layer1_size

        self.encoder1 = nn.Linear(input_size, layer1_size)
	self.decoder1 = nn.Linear(layer1_size, input_size)

    def forward(self, x):
        x = F.tanh(self.encoder1(x))
	x = self.decoder1(x)
        return x


class AutoEncoder_2(nn.Module):
    def __init__(self, input_size=48, layer1_size=20, layer2_size=10):
        super(AutoEncoder_2, self).__init__()

	self.input_size = input_size       
	self.layer1_size = layer1_size
	self.layer2_size = layer2_size

        self.encoder1 = nn.Linear(input_size, layer1_size)
	self.encoder2 = nn.Linear(layer1_size, layer2_size)
        self.decoder2 = nn.Linear(layer2_size, layer1_size)
	self.decoder1 = nn.Linear(layer1_size, input_size)

    def forward(self, x):
        x = F.tanh(self.encoder1(x))
	x = F.tanh(self.encoder2(x))
        x = F.tanh(self.decoder2(x))
	x = self.decoder1(x)
        return x


class AutoEncoder_3(nn.Module):
    def __init__(self, input_size=48, layer1_size=20, layer2_size=10, layer3_size=5):
        super(AutoEncoder_3, self).__init__()

	self.input_size = input_size       
	self.layer1_size = layer1_size
	self.layer2_size = layer2_size
	self.layer3_size = layer3_size

        self.encoder1 = nn.Linear(input_size, layer1_size)
	self.encoder2 = nn.Linear(layer1_size, layer2_size)
	self.encoder3 = nn.Linear(layer2_size, layer3_size)
        self.decoder3 = nn.Linear(layer3_size, layer2_size)
        self.decoder2 = nn.Linear(layer2_size, layer1_size)
	self.decoder1 = nn.Linear(layer1_size, input_size)

    def forward(self, x):
        x = F.tanh(self.encoder1(x))
	x = F.tanh(self.encoder2(x))
	x = F.tanh(self.encoder3(x))
        x = F.tanh(self.decoder3(x))
        x = F.tanh(self.decoder2(x))
	x = self.decoder1(x)
        return x


class Express_encode(nn.Module):
    def __init__(self, input_size=48, layer1_size=20, layer2_size=10, layer3_size=5):
        super(ExpressCode, self).__init__()

	self.input_size = input_size       
	self.layer1_size = layer1_size
	self.layer2_size = layer2_size
	self.layer3_size = layer3_size

        self.encoder1 = nn.Linear(input_size, layer1_size)
	self.encoder2 = nn.Linear(layer1_size, layer2_size)
	self.encoder3 = nn.Linear(layer2_size, layer3_size)

    def forward(self, x):
        x = F.tanh(self.encoder1(x))
	x = F.tanh(self.encoder2(x))
	x = F.tanh(self.encoder3(x))
        return x
