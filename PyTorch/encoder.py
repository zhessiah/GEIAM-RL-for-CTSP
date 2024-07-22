import torch
import torch.nn as nn
# from torchsummary import summary

from layers import MultiHeadAttention
from data import generate_data
import math

class Normalization(nn.Module):

	def __init__(self, embed_dim, normalization = 'batch'):
		super().__init__()

		normalizer_class = {
			'batch': nn.BatchNorm1d,
			'instance': nn.InstanceNorm1d}.get(normalization, None)
		self.normalizer = normalizer_class(embed_dim, affine=True)
		# Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
	# 	self.init_parameters()

	# def init_parameters(self):
	# 	for name, param in self.named_parameters():
	# 		stdv = 1. / math.sqrt(param.size(-1))
	# 		param.data.uniform_(-stdv, stdv)

	def forward(self, x):

		if isinstance(self.normalizer, nn.BatchNorm1d):
			# (batch, num_features)
			# https://discuss.pytorch.org/t/batch-normalization-of-linear-layers/20989
			return self.normalizer(x.view(-1, x.size(-1))).view(*x.size())
		
		elif isinstance(self.normalizer, nn.InstanceNorm1d):
			return self.normalizer(x.permute(0, 2, 1)).permute(0, 2, 1)
		else:
			assert self.normalizer is None, "Unknown normalizer type"
			return x



class Gating(torch.nn.Module):
    def __init__(self, d_input=192, bg=0.1):
        super(Gating, self).__init__()
        self.Wr = torch.nn.Linear(d_input, d_input)
        self.Ur = torch.nn.Linear(d_input, d_input)
        self.Wz = torch.nn.Linear(d_input, d_input)
        self.Uz = torch.nn.Linear(d_input, d_input)
        self.Wg = torch.nn.Linear(d_input, d_input)
        self.Ug = torch.nn.Linear(d_input, d_input)
        self.bg = bg

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, x, y):
        r = self.sigmoid(self.Wr(y) + self.Ur(x))
        z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h = self.tanh(self.Wg(y) + self.Ug(torch.mul(r, x)))
        g = torch.mul(1 - z, x) + torch.mul(z, h)
        return g


class ResidualBlock_BN(nn.Module):
	def __init__(self, MHA, BN, GATE=None, **kwargs):
		super().__init__(**kwargs)
		self.MHA = MHA
		self.BN = BN
		self.GATE = GATE

	def forward(self, x, colors, mask = None):
		if mask is None:
			return self.BN(self.GATE(x, self.MHA(x, colors))) #TODO: GATE abalation

		return self.BN(self.GATE(x, self.MHA(x, colors, mask)))




class ResidualBlock_BN_2(nn.Module):
	def __init__(self, FF, BN, GATE=None, **kwargs):
		super().__init__(**kwargs)
		self.FF = FF
		self.BN = BN	
		self.GATE = GATE

	def forward(self, x, mask = None):
		if mask is None:
			return self.BN(self.GATE(x, self.FF(x))) #TODO: GATE abalation
			# return self.BN(x + self.FF(x))
		return self.BN(self.GATE(x, self.FF(x, mask)))

class SelfAttention(nn.Module):
	def __init__(self, MHA, **kwargs):
		super().__init__(**kwargs)
		self.MHA = MHA

	def forward(self, x, colors, mask = None):
		return self.MHA([x, x, x], colors, mask = mask)

class EncoderLayer(nn.Module):
	# nn.Sequential):
	def __init__(self, n_heads = 8, FF_hidden = 512, embed_dim = 128, **kwargs):
		super().__init__(**kwargs)
		self.n_heads = n_heads
		self.FF_hidden = FF_hidden
		self.BN1 = Normalization(embed_dim, normalization = 'batch')
		self.BN2 = Normalization(embed_dim, normalization = 'batch')

		# self.MHA_sublayer = ResidualBlock_BN(
		# 		SelfAttention(
		# 			MultiHeadAttention(n_heads = self.n_heads, embed_dim = embed_dim, need_W = True)
		# 		),
		# 	self.BN1
		# 	)
  
		self.MHA_sublayer = ResidualBlock_BN(
				SelfAttention(
					MultiHeadAttention(n_heads = self.n_heads, embed_dim = embed_dim, need_W = True)
				),
			self.BN1,
			Gating(embed_dim)#TODO: GATE abalation
			)

		self.FF_sublayer = ResidualBlock_BN_2(
			nn.Sequential(
					nn.Linear(embed_dim, FF_hidden, bias = True),
					nn.ReLU(),
					nn.Linear(FF_hidden, embed_dim, bias = True)
			),
			self.BN2,
			Gating(embed_dim) #TODO: GATE abalation
		)
		
	def forward(self, x, colors, mask=None):
		"""	arg x: (batch, n_nodes, embed_dim)
			return: (batch, n_nodes, embed_dim)
		"""
		return self.FF_sublayer(self.MHA_sublayer(x, colors, mask = mask))
		
class GraphAttentionEncoder(nn.Module):
	def __init__(self, embed_dim = 128, n_heads = 8, n_layers = 3, FF_hidden = 512):
		super().__init__()
		self.init_W_depot = torch.nn.Linear(2, embed_dim, bias = True)
		self.init_W = torch.nn.Linear(2, embed_dim, bias = True)
		self.encoder_layers = nn.ModuleList([EncoderLayer(n_heads, FF_hidden, embed_dim) for _ in range(n_layers)])
	
	def forward(self, x, mask = None):
		""" x[0] -- depot_xy: (batch, 2) --> embed_depot_xy: (batch, embed_dim)
			x[1] -- customer_xy: (batch, n_nodes, 2)
			--> concated_customer_feature: (batch, n_nodes-1, 3) --> embed_customer_feature: (batch, n_nodes-1, embed_dim)
			embed_x(batch, n_nodes, embed_dim)
			x[2] -- colors: (batch, n_nodes) values in {0, 1, 2, 3}

			return: (node embeddings(= embedding for all nodes), graph embedding(= mean of node embeddings for graph))
				=((batch, n_nodes, embed_dim), (batch, embed_dim))
		"""
		colors = x[2]
		x = torch.cat([self.init_W_depot(x[0])[:, None, :],  #(batch, 2) -> (batch, embed_dim) ->   (batch, 1, embed_dim)
				self.init_W(x[1])], dim = 1) # (batch, n_customer, 2) -> (batch, n_customer, embed_dim)
		
		# x.shape: (batch, n_nodes = (n_customer + 1), embed_dim)
	
		for layer in self.encoder_layers:
			x = layer(x, colors, mask)

		return (x, torch.mean(x, dim = 1))

if __name__ == '__main__':
	batch = 5
	n_nodes = 21
	encoder = GraphAttentionEncoder(n_layers = 1)
 
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	data = generate_data(device, n_samples = batch, n_customer = n_nodes-1)
	# mask = torch.zeros((batch, n_nodes, 1), dtype = bool)
	output = encoder(data, mask = None)
	print('output[0].shape:', output[0].size())
	print('output[1].shape', output[1].size())
	
	# summary(encoder, [(2), (20,2), (20)])
	cnt = 0
	for i, k in encoder.state_dict().items():
		print(i, k.size(), torch.numel(k))
		cnt += torch.numel(k)
	print(cnt)

	# output[0].mean().backward()
	# print(encoder.init_W_depot.weight.grad)

