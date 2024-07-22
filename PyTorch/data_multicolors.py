import torch
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

# multi colors
def generate_data(device, n_samples=10, n_customer=20, seed=None):
	""" 
	x[0] -- depot_xy: (batch, 2)
	x[1] -- customer_xy: (batch, n_nodes-1(n_customer), 2)
	x[2] -- demand: (batch, n_nodes-1(n_customer), 4) bool (multicolor. 4 colors:true, true, true, true)
	"""
	if seed is not None:
		torch.manual_seed(seed)


	return (
		torch.rand((n_samples, 2), device=device),
		torch.rand((n_samples, n_customer, 2), device=device),
		torch.randint(1, 15, (n_samples, n_customer), device=device), # TODO: change to 4-digit binary:0000
	)


class Generator(Dataset):
    
	def __init__(self, device, n_samples = 5120, n_customer = 20):
		self.tuple = generate_data(device, n_samples, n_customer)
		print('Generator: ', self.tuple[0].size(), self.tuple[1].size(), self.tuple[2].size())

	def __getitem__(self, idx):
		return (self.tuple[0][idx], self.tuple[1][idx], self.tuple[2][idx])

	def __len__(self):
		return self.tuple[0].size(0)



if __name__ == '__main__':
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print('device-->', device)
	
	data = generate_data(device, n_samples = 128, n_customer = 20, seed = 123)
	for i in range(3):
		print(data[i].dtype)
		print(data[i].shape)
	
	batch, batch_steps, n_customer = 128, 100000, 20
	dataset = Generator(device, n_samples = batch*batch_steps, n_customer = n_customer)
	data = next(iter(dataset))	
	
	dataloader = DataLoader(dataset, batch_size = 128, shuffle = True)
	print('use datalodaer ...')
	for i, data in enumerate(dataloader):
		for j in range(len(data)):
			print(data[j].dtype)# torch.float32
			print(data[j].size())	
		if i == 0:
			break