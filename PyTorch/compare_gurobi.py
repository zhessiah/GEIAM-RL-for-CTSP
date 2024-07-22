from torch.utils.data import DataLoader
import torch
from matplotlib import pyplot as plt
import numpy as np
import math
import os
from model import AttentionModel
from data_from_npz import generate_data
from ctsp_plot import draw_path
import time


def write_sol_to_txt(routes, num_nodes):
    with open("{}.txt".format(num_nodes), "w") as file:
        for array in routes:
            numbers = " ".join(str(num) for num in array)
            file.write(numbers + "\n")


if __name__ == "__main__":
    
    model = AttentionModel(embed_dim = 192, n_encode_layers = 3, n_heads = 8, tanh_clipping = 10., FF_hidden = 512)
    
    model.eval()
    model = model.to(torch.device('cuda:0'))
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load('PyTorch/Weights/VRP50_train_best_model_0619_00_03.pt'))

    n_customer = 100
    dataset = generate_data(device = torch.device('cuda:0'), n_customer=n_customer)
    # Run the model
    model.eval()
    t0 = time.time()
    min_cost = 100000
    min_pi = None
    min_index = None
    with torch.no_grad():
        for i in range(1):
            cost, _, pi = model(dataset, return_pi = True, decode_type = "greedy") # same time, explore best solution
            if min(cost) < min_cost:
                min_cost = min(cost)
                min_pi = pi
                # get the index of min_cost
                min_index = np.argmin(cost.cpu().numpy())
        # cost, _, pi = model(dataset, return_pi = True, decode_type = "sampling")
    print((time.time() - t0) / 20)
    print(min_cost)

    tours = min_pi
    
    
    # Plot the results
    for i, (data, tour, depot) in enumerate(zip(dataset[1], tours, dataset[0])):
        if i == min_index:
            routes = [r[r != 0] for r in np.split(tour.cpu().numpy(), np.where(tour.cpu() == 0)[0]) if (r != 0).any()]
            write_sol_to_txt(routes, n_customer)
            
            locs = data.cpu().numpy()
            depot = depot.cpu().numpy()
            draw_path(routes, locs, depot, 1)
            
        