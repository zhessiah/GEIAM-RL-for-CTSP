import torch
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

# single color
def generate_data(device, n_samples=10, n_customer=20, seed=4321):
    
	""" https://pytorch.org/docs/master/torch.html?highlight=rand#torch.randn
	x[0] -- depot_xy: (n_samples, 2)
	x[1] -- customer_xy: (n_samples, n_customer, 2)
	x[2] -- colors: (batch, n_nodes-1)
	"""
	if seed is not None:
		torch.manual_seed(seed)
		np.random.seed(seed)

	return (
		torch.rand((n_samples, 2), device=device), # depot_xy
		torch.rand((n_samples, n_customer, 2), device=device), # customer_xy
		torch.randint(1, 6, (n_samples, n_customer), device=device), # colors, [i,j] represents the color range
	)


class Generator(Dataset):
	""" https://github.com/utkuozbulak/pytorch-custom-dataset-examples
		https://github.com/wouterkool/attention-learn-to-route/blob/master/problems/vrp/problem_vrp.py
		https://github.com/nperlmut31/Vehicle-Routing-Problem/blob/master/dataloader.py
	"""
	def __init__(self, device, n_samples = 5120, n_customer = 20):
		self.tuple = generate_data(device, n_samples, n_customer)
		print('Generator: ', self.tuple[0].size(), self.tuple[1].size(), self.tuple[2].size())

	def __getitem__(self, idx):
		return (self.tuple[0][idx], self.tuple[1][idx], self.tuple[2][idx])

	def __len__(self):
		return self.tuple[0].size(0)






# TSPLIB datasets
def generate_data_from_tspfile(file_path, device, seed = None, samples = 1):
	data = np.loadtxt(file_path, dtype=int)
	customer_xy = torch.tensor(data[:, 1:], device=device).float()
	n_customer = customer_xy.shape[0]
	if seed is not None:
		torch.manual_seed(seed)
	depot_idx = torch.randint(0, n_customer, (1,), device=device)
	depot_xy = customer_xy[depot_idx].repeat(samples, 1) # (samples, 2)
	return (
		depot_xy, # depot_xy
		(customer_xy.unsqueeze(0)).repeat(samples, 1, 1), # customer_xy
		torch.randint(1, 5, (1, n_customer), device=device).repeat(samples, 1), # colors
	) 

class Generator_TSPLIB(Dataset): # TODO: all TSPLIB files in one generator
	def __init__(self, device, Path):
		self.tuple = generate_data_from_tspfile(Path, device)
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
