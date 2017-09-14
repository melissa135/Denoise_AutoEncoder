import os
import torch
import collections
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from define_network import Compression_encoder,AutoEncoder_3
from sample_set import Sample_set

if __name__ == '__main__':

    path_ = os.path.abspath('.')

    if not os.path.exists(path_+'/vision') :
        os.mkdir(path_+'/vision')

    n = 100

    fname = path_ + '/autoencoder_layer3.pth'
    ae = AutoEncoder_3()
    ae.load_state_dict(torch.load(fname))

    fname = path_ + '/compression_encoder.pth'
    ce = Compression_encoder()
    ce.load_state_dict(torch.load(fname))
    
    testset = Sample_set(path_+'/test')
    testloader = torch.utils.data.DataLoader(testset,batch_size=1,shuffle=False)

    for i,data in enumerate(testloader,0):

        input,target = data
        input = Variable(input)
        actual = target[0].numpy()

        output = ae(input.float())
        output = output.data[0].numpy()
	min_ = min(actual)
	max_ = max(actual)
	#print actual,output
	
	code = ce(input.float())
	code = code.data[0].numpy()

        X = range(0,len(actual))
        plt.figure(figsize=(12,8),dpi=80)
        plt.plot(X,actual,color='black',linewidth=1,label='original')
        plt.plot(X,output,color='red',linewidth=1,label='encoding_recover')
        plt.legend(loc='upper right', frameon=False, fontsize=20)
	plt.text(5,min_+(max_-min_)*0.7,code[0],fontsize=20)
	plt.text(5,min_+(max_-min_)*0.6,code[1],fontsize=20)
	plt.text(5,min_+(max_-min_)*0.5,code[2],fontsize=20)
	plt.text(5,min_+(max_-min_)*0.4,code[3],fontsize=20)
	plt.text(5,min_+(max_-min_)*0.3,code[4],fontsize=20)
        plt.savefig(path_+'/vision/vision_%d.png'%i)
        #plt.show()
	if i == n :
	    break
