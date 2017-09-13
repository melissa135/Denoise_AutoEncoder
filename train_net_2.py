import os.path
import torch
import collections
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from define_network import AutoEncoder_1,AutoEncoder_2
from sample_set import Sample_set
from torch.autograd import Variable


def test_loss(ae,testloader):

    total_loss = 0
    criterion_ = nn.MSELoss()
    for i,data in enumerate(testloader,0):

        input,target = data
        input,target = Variable(input),Variable(target)

        output = ae(input.float())
        loss = criterion_(output, target.float())# + penalty
        total_loss += loss.data[0]

    return total_loss


if __name__ == '__main__':

    path_ = os.path.abspath('.')

    batchsize = 8

    trainset = Sample_set(path_+'/train')
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=batchsize,shuffle=True,num_workers=2)

    testset = Sample_set(path_+'/test')
    testloader = torch.utils.data.DataLoader(testset,batch_size=batchsize,shuffle=True,num_workers=2)

    print 'Training AutoEncoder.'
        
    max_epochs = 100

    ae2 = AutoEncoder_2()
    print ae2
    
    # load the pretrain net
    ae1 = AutoEncoder_1()
    fname = path_ + '/autoencoder_layer1.pth'
    ae1.load_state_dict(torch.load(fname))

    new_dict = collections.OrderedDict()
    for key in ae1.state_dict().keys():
	new_dict[key] = ae1.state_dict()[key]
	
    new_dict['encoder2.weight'] = ae2.state_dict()['encoder2.weight']
    new_dict['encoder2.bias'] = ae2.state_dict()['encoder2.bias']
    new_dict['decoder2.weight'] = ae2.state_dict()['decoder2.weight']
    new_dict['decoder2.bias'] = ae2.state_dict()['decoder2.bias']

    ae2.load_state_dict(new_dict)
   
    # set the fixed parameters
    for p in ae2.encoder1.parameters():
	p.requires_grad = False
    for p in ae2.decoder1.parameters():
	p.requires_grad = False

    optimizer = optim.Adam([{'params':ae2.encoder2.parameters()},
			    {'params':ae2.decoder2.parameters()}],lr=0.001)
    #optimizer = optim.Adam(ae2.parameters(),lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(0, max_epochs):

        current_loss = 0
        for i,data in enumerate(trainloader,0):

            input,target = data
            input,target = Variable(input),Variable(target)

            ae2.zero_grad()

            output = ae2(input.float())
            loss = criterion(output, target.float())# + penalty
      
            loss.backward()
            optimizer.step()

            loss = loss.data[0]
            current_loss += loss
	
	t_loss = test_loss(ae2,testloader)
        print ( '[ %d ] loss : %.4f %.4f' % \
	      ( epoch+1, batchsize*current_loss/trainset.__len__(), batchsize*t_loss/testset.__len__()) )

        current_loss = 0

    torch.save(ae2.state_dict(),path_+'/autoencoder_layer2.pth')

