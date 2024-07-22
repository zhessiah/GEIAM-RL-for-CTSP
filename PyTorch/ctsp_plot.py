from torch.utils.data import DataLoader
import torch
from matplotlib import pyplot as plt
import numpy as np
import math
import os
from model import AttentionModel
# from data import generate_data
from data_from_npz import generate_data

def draw_path(car_routes, CityCoordinates, depot, ind):
    # 画路径图
    # 输入：line-路径，CityCoordinates-城市坐标；
    # 输出：路径图
    totaldis = 0
    vehic_ind = 0
    lines = []
    _labels = []
    for route in car_routes:
        routedis = 0
        x, y = [], []
        x.append(depot[0])
        y.append(depot[1])
        x_prev = depot[0]
        y_prev = depot[1]
        for i in route:
            Coordinate = CityCoordinates[i - 1]
            x.append(Coordinate[0])
            y.append(Coordinate[1])
            routedis += np.sqrt((Coordinate[0] - x_prev) ** 2 + (Coordinate[1] - y_prev) ** 2)
            x_prev = Coordinate[0]
            y_prev = Coordinate[1]
        x.append(depot[0])
        y.append(depot[1])
        
        # 总距离
        routedis += math.sqrt((depot[0] - x_prev) ** 2 + (depot[1] - y_prev) ** 2)
        totaldis += routedis
        
        # for legend
        if vehic_ind == 0:
            line, = plt.plot(x, y, 'o-', alpha=0.8, linewidth=0.8, color = 'red')
        elif vehic_ind == 1:
            line, = plt.plot(x, y, 'o-', alpha=0.8, linewidth=0.8, color = 'orange')
        elif vehic_ind == 2:
            line, = plt.plot(x, y, 'o-', alpha=0.8, linewidth=0.8, color = 'blue')
        else:
            line, = plt.plot(x, y, 'o-', alpha=0.8, linewidth=0.8, color = 'green')
        lines.append(line)
        _labels.append('salesman' + str(vehic_ind))
        vehic_ind += 1
        



    plt.scatter(depot[0], depot[1], s=800, color = 'red',
                marker='*')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('{} nodes, total length {:.2f}'.format(CityCoordinates.shape[0], totaldis))
    plt.legend(handles=lines, labels=_labels, loc='lower right')
    plt.savefig(os.path.join('./images', 'ctsp_{}.pdf'.format(ind)))
    plt.show()
    plt.close()
    

def draw_path_without_depot(car_routes, CityCoordinates, ind):
    # 画路径图
    # 输入：line-路径，CityCoordinates-城市坐标；
    # 输出：路径图
    totaldis = 0
    vehic_ind = 0
    lines = []
    _labels = []
    for route in car_routes:
        routedis = 0
        x, y = [], []
        depot = CityCoordinates[route[0] - 1]
        x.append(depot[0])
        y.append(depot[1])
        x_prev = depot[0]
        y_prev = depot[1]
        for i in range(1, len(route)):
            Coordinate = CityCoordinates[route[i] - 1]
            x.append(Coordinate[0])
            y.append(Coordinate[1])
            routedis += np.sqrt((Coordinate[0] - x_prev) ** 2 + (Coordinate[1] - y_prev) ** 2)
            x_prev = Coordinate[0]
            y_prev = Coordinate[1]
        x.append(depot[0])
        y.append(depot[1])
        
        # 总距离
        routedis += math.sqrt((depot[0] - x_prev) ** 2 + (depot[1] - y_prev) ** 2)
        totaldis += routedis
        plt.scatter(depot[0], depot[1], s=800, color = 'red',
                marker='*')
        
        # for legend
        line, = plt.plot(x, y, 'o-', alpha=0.8, linewidth=0.8)
        lines.append(line)
        _labels.append('salesman' + str(vehic_ind))
        vehic_ind += 1
        


    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('{} nodes, total length {:.2f}'.format(CityCoordinates.shape[0], totaldis))
    plt.legend(handles=lines, labels=_labels, loc='lower right')
    plt.savefig(os.path.join('./images', 'ctsp_{}.pdf'.format(ind)))
    plt.show()
    plt.close()


def regular_process(dataset):
    
    n_samples = dataset[0].shape[0]
    for i in range(n_samples):
        dataset[0][i][0] = 0.5
        dataset[0][i][1] = 0.5
        depotx = dataset[0][i][0]
        depoty = dataset[0][i][1]
        for j in range(dataset[1][i].shape[0]):
            nowx = dataset[1][i][j][0]
            nowy = dataset[1][i][j][1]
            if nowx < depotx and nowy < depoty:
                dataset[2][i][j] = 1
            elif nowx < depotx and nowy > depoty:
                dataset[2][i][j] = 2
            elif nowx > depotx and nowy > depoty:
                dataset[2][i][j] = 3
            else :
                dataset[2][i][j] = 4
    return dataset


if __name__ == "__main__":
    
    model = AttentionModel(embed_dim = 192, n_encode_layers = 3, n_heads = 8, tanh_clipping = 10., FF_hidden = 512)
    
    model.eval()
    model = model.to(torch.device('cuda:0'))
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load('Weights/VRP70_train_best_model_0525_00_03.pt'))

    dataset = generate_data(device = torch.device('cuda:0'), n_samples = 20, n_customer = 100, seed = 203)
    dataset = regular_process(dataset)
    # Run the model
    model.eval()
    with torch.no_grad():
        cost, _, pi = model(dataset, return_pi = True, decode_type = "sampling")
        # costs.append(cost)
    print(min(cost))

    tours = pi
    
    
    # Plot the results
    for i, (data, tour, depot) in enumerate(zip(dataset[1], tours, dataset[0])):
        routes = [r[r != 0] for r in np.split(tour.cpu().numpy(), np.where(tour.cpu() == 0)[0]) if (r != 0).any()]
        locs = data.cpu().numpy()
        depot = depot.cpu().numpy()
        # draw_path(routes, locs, depot, i)
        draw_path_without_depot(routes, locs, i)