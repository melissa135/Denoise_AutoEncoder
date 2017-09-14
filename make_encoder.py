import os.path
import torch
import collections
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from define_network import Compression_encoder,AutoEncoder_3
from sample_set import Sample_set

if __name__ == '__main__':

    path_ = os.path.abspath('.')

    fname = path_ + '/autoencoder_layer3.pth'
    ae = AutoEncoder_3()
    ae.load_state_dict(torch.load(fname))
    
    ce = Compression_encoder()
    new_dict = collections.OrderedDict()
    new_dict['encoder1.weight'] = ae.state_dict()['encoder1.weight']
    new_dict['encoder1.bias'] = ae.state_dict()['encoder1.bias']
    new_dict['encoder2.weight'] = ae.state_dict()['encoder2.weight']
    new_dict['encoder2.bias'] = ae.state_dict()['encoder2.bias']
    new_dict['encoder3.weight'] = ae.state_dict()['encoder3.weight']
    new_dict['encoder3.bias'] = ae.state_dict()['encoder3.bias']
    ce.load_state_dict(new_dict)

    torch.save(ce.state_dict(),path_+'/compression_encoder.pth')
