import torch
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

# single color
def generate_data(device, n_samples = 20, n_customer = 100, seed = 23):

    """ https://pytorch.org/docs/master/torch.html?highlight=rand#torch.randn
    x[0] -- depot_xy: (n_samples, 2)
    x[1] -- customer_xy: (n_samples, n_customer, 2)
    x[2] -- colors: (batch, n_nodes-1)
    """
    data = np.load('PyTorch/npdata_random/arrays%s.npz'%(n_customer)) # TODO: np random or regular
    depot_xy = torch.tensor(data['depot'], device=device)
    customer_xy = torch.tensor(data['cities'], device=device)
    colors = torch.tensor(data['city_colors'], device=device)
    
    depot_xy = depot_xy.repeat(n_samples, 1)
    customer_xy = customer_xy.repeat(n_samples, 1, 1)
    colors = colors.repeat(n_samples, 1)
    
    depot_xy = depot_xy.to(torch.float32)
    customer_xy = customer_xy.to(torch.float32)
    colors = colors.to(torch.float32)


    # Return a tuple containing the three tensors
    return (depot_xy, customer_xy, colors)