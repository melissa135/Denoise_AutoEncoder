import os
import torch
import torch.utils.data as data
from pandas.io.parsers import read_csv


def read_files(folder):

    length = 48
    
    daily_data = []

    for root, _, fnames in os.walk(folder):
        for fname in fnames:
            
            path = os.path.join(root, fname)
            df = read_csv(path)
            
            for i in range(0,len(df)):
		if i % length == 0 :
		    if i != 0 :
		        daily_data.append(temp[:])
		    temp = [ df['price_change'][i] ]
		else :
		    temp.append(df['price_change'][i])

	    #daily_data.append(temp[:])	

    return daily_data


class Sample_set(data.Dataset):

    def __init__(self, folder):

	data = read_files(folder)
	print 'This set contains %d items.' % len(data)
	self.data = data

    def __getitem__(self, index):

	item = torch.Tensor(self.data[index][:])
        target = torch.Tensor(self.data[index][:])
	return item,target

    def __len__(self):
        return len(self.data)


class Sample_set_2(data.Dataset):

    def __init__(self, folder):

	data = read_files(folder)
	print 'This set contains %d items.' % len(data)
	self.data = data

    def __getitem__(self, index):

	item = torch.Tensor(self.data[index][:])
        target = torch.Tensor([ self.data[index+1][-1] ])
	return item,target

    def __len__(self):
        return len(self.data)-1
