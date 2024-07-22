from data import generate_data_from_tspfile
# from data_generatein100 import generate_data
# from data_multicolors import generate_data
from data_from_npz import generate_data
from model import AttentionModel
import torch
from time import time

if __name__ == '__main__':
    
    
    initial_mem = torch.cuda.memory_allocated()
    
    model = AttentionModel(embed_dim = 192, n_encode_layers = 3, n_heads = 8, tanh_clipping = 10., FF_hidden = 512)
    
    model.eval()
    model = model.to(torch.device('cuda:0'))
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load('PyTorch/Weights/5-colors/VRP100_train_best_model_0501_17_52.pt'))
    print(model)

    Path = 'PyTorch/TSPLIB/eli101.txt'
    # dataset = generate_data_from_tspfile(Path, device = torch.device('cuda:0'), seed=123, samples=10)
    dataset = generate_data(device = torch.device('cuda:0'), n_samples = 10, n_customer = 100, seed = 123)
    # samples: num of copies
    
    t0 = time()
    # costs = []
    # for i in range(10):
    with torch.no_grad():
        cost, _, pi = model(dataset, return_pi = True, decode_type = "sampling")
        # costs.append(cost)
    print("totol time: ", time() - t0)
    final_mem = torch.cuda.memory_allocated()
    mem_used = final_mem - initial_mem
    print(f'Memory used in GB: {mem_used / 1024**3}')
    max_mem_used = torch.cuda.max_memory_allocated()
    print(f'Max memory used in GB: {max_mem_used / 1024**3}')
    
    print(min(cost))
    
    # output[1].mean().backward()
    # print(model.Decoder.Wout.weight.grad)
    # print(model.Encoder.init_W_depot.weight.grad)