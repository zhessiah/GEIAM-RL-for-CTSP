from data import generate_data_from_tspfile
import unittest
import torch
import os

class Test(unittest.TestCase):
    def test_generate_data_from_tspfile(self):
        device = torch.device('cuda:0')
        print(os.getcwd())
        file_path = 'PyTorch/TSPLIB/eli101.txt'
        depot_xy, customer_xy, colors = generate_data_from_tspfile(file_path, device)
        print('depot_xy: ', depot_xy.shape)
        print('customer_xy: ', customer_xy.shape)
        print('colors: ', colors.shape)
        
if __name__ == '__main__':
    a = Test()
    a.test_generate_data_from_tspfile()