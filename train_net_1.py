import os.path
import torch
import collections
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from define_network import AutoEncoder
from sample_set import Sample_set
from torch.autograd import Variable


def test_loss(ae,testloader):

    total_loss = 0
    criterion_ = nn.MSELoss()
    for i,data in enumerate(testloader,0):

        input,target = data
        input,target = Variable(input),Variable(target)

        output = ae(input.float())
        loss = criterion_(output, target.float())
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
    
    ae = AutoEncoder()
    print ae

    optimizer = optim.Adam(ae.parameters(),lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(0, max_epochs):

        current_loss = 0
        for i,data in enumerate(trainloader,0):

            input,target = data
            input,target = Variable(input),Variable(target)

            ae.zero_grad()

            output = ae(input.float())
            loss = criterion(output, target.float())
      
            loss.backward()
            optimizer.step()

            loss = loss.data[0]
            current_loss += loss
	
	t_loss = test_loss(ae,testloader)
        print ( '[ %d ] loss : %.4f %.4f' % \
	      ( epoch+1, batchsize*current_loss/trainset.__len__(), batchsize*t_loss/testset.__len__()) )

        current_loss = 0

    torch.save(ae.state_dict(),path_+'/autoencoder_layer1.pth')

