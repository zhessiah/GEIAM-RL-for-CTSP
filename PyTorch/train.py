import torch
import multiprocessing
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time
import math
from tensorboard_logger import Logger as TbLogger
import os
from datetime import datetime

from model import AttentionModel
from baseline import RolloutBaseline
from data import generate_data, Generator
from config import Config, load_pkl, train_parser


def train(cfg, log_path = None):
	torch.multiprocessing.set_start_method('spawn')
	torch.backends.cudnn.benchmark = True
	def rein_loss(model, inputs, bs, t, device):
		inputs = list(map(lambda x: x.to(device), inputs))
		L, ll = model(inputs, decode_type = 'sampling')  # train with sampling
		b = bs[t] if bs is not None else baseline.eval(inputs, L)
		return ((L - b.to(device)) * ll).mean(), L.mean()
	
	
	# model
	model = AttentionModel(cfg.embed_dim, cfg.n_encode_layers, cfg.n_heads, cfg.tanh_clipping)
	model.train()
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model.to(device)
	if torch.cuda.device_count() >= 1:
		model = torch.nn.DataParallel(model)
	
	# tensorboard_logger
	tb_logger = TbLogger(os.path.join('tensorboard_log_dir', str(cfg.n_customer), datetime.now().strftime('%m%d_%H_%M')))
	
	
	# checkpoint: optional
	# model.load_state_dict(torch.load('Weights/VRP70_train_best_model_0525_00_03.pt')) # TODO:change path
	
	
	baseline = RolloutBaseline(model, cfg.task, cfg.weight_dir, cfg.n_rollout_samples, 
								cfg.embed_dim, cfg.n_customer, cfg.warmup_beta, cfg.wp_epochs, device)
	optimizer = optim.Adam(model.parameters(), lr = cfg.lr)
	
	t1 = time()
	
	best_reward = 1000 # TODO: here need to be changed after each training process if you want to continue training
	
	val_dataset = generate_data(device, n_samples = 100, n_customer = 400) # validation dataset #TODO: change to 100 nodes
	
	
	for epoch in range(cfg.epochs):
		ave_loss, ave_L = 0., 0.
		dataset = Generator(device, cfg.batch*cfg.batch_steps, cfg.n_customer)
		
		# bs = baseline.eval_all(dataset) 
		# bs = bs.view(-1, cfg.batch) if bs is not None else None# bs: (cfg.batch_steps, cfg.batch) or None
		bs = None
		dataloader = DataLoader(dataset, batch_size = cfg.batch, shuffle = True, num_workers=3)
		
		
		# tensorboard_logger
		step = epoch * cfg.batch_steps
		if tb_logger is not None:
			tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

		for t, inputs in enumerate(dataloader):
			loss, L_mean = rein_loss(model, inputs, bs, t, device)
			optimizer.zero_grad()
			loss.backward()
			# print('grad: ', model.Decoder.Wk1.weight.grad[0][0])
			# https://github.com/wouterkool/attention-learn-to-route/blob/master/train.py
			grad_norms = nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0, norm_type = 2)
			optimizer.step()
			
			
			ave_loss += loss.item()
			ave_L += L_mean.item()
			
			if t%(cfg.batch_verbose) == 0:
				t2 = time()
				print('Epoch %d (batch = %d): Loss: %1.3f L: %1.3f, %dmin%dsec'%(
					epoch, t, ave_loss/(t+1), ave_L/(t+1), (t2-t1)//60, (t2-t1)%60))
				if cfg.islogger:
					if log_path is None:
						log_path = '%s%s_%s.csv'%(cfg.log_dir, cfg.task, cfg.dump_date)#cfg.log_dir = ./Csv/
						with open(log_path, 'w') as f:
							f.write('time,epoch,batch,loss,cost\n')
					with open(log_path, 'a') as f:
						f.write('%dmin%dsec,%d,%d,%1.3f,%1.3f\n'%(
							(t2-t1)//60, (t2-t1)%60, epoch, t, ave_loss/(t+1), ave_L/(t+1)))
				t1 = time()


		
		# update baseline
		baseline.epoch_callback(model, epoch)
  
		# save new best model
		if ave_L/(t+1) < best_reward:
			best_reward = ave_L/(t+1)
			print("new best model finded:" + str(best_reward)) 
			torch.save(model.state_dict(), '%s%s_best_model_%s.pt'%(cfg.weight_dir, cfg.task, cfg.dump_date))

		# validate model
		avg_cost, best_cost = validate(model, val_dataset)
  
  
		# tensorboard log
		if tb_logger is not None:
			tb_logger.log_value('avg_value', ave_L/(t+1), step)
			tb_logger.log_value('avg_loss', ave_loss/(t+1), step)
			tb_logger.log_value('avg_cost', avg_cost, step)
			tb_logger.log_value('best_cost', best_cost, step)
			tb_logger.log_value('grad_norm', grad_norms, step)

		step += 1


def validate(model, dataset):
	# Validate
	print('Validating...')
	model.eval()
	
	with torch.no_grad():
		cost, _, = model(dataset, return_pi = False, decode_type = "sampling")
	
	avg_cost = cost.mean()
	best_cost = cost.min()
	print('Validation overall avg_cost: {} +- {}, best_cost: {}'.format(
		avg_cost, torch.std(cost) / math.sqrt(len(cost)), best_cost))
 
	model.train()

	return avg_cost, best_cost



if __name__ == '__main__':
    
	cfg = load_pkl(train_parser().path)
	train(cfg)	
