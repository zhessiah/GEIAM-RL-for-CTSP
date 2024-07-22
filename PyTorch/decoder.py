import torch
import torch.nn as nn

from layers import MultiHeadAttention, DotProductAttention
from data import generate_data
from decoder_utils import TopKSampler, CategoricalSampler, Env, Env_multicolors

class DecoderCell(nn.Module):
	def __init__(self, embed_dim = 128, n_heads = 8, clip = 10., multicolor = False, **kwargs): #TODO: multicolor is changed here, when you have to plot ctsp
		super().__init__(**kwargs)
		
  
		# static linear projection
		self.Wk1 = nn.Linear(embed_dim, embed_dim, bias = False)
		self.Wv = nn.Linear(embed_dim, embed_dim, bias = False)
		self.Wk2 = nn.Linear(embed_dim, embed_dim, bias = False)
		self.Wq_fixed = nn.Linear(embed_dim, embed_dim, bias = False) 
  
  
		# dynamic linear projection
		self.Wout = nn.Linear(embed_dim, embed_dim, bias = False)
		self.Wq_step = nn.Linear(embed_dim+1, embed_dim, bias = False)
  
		# dynamic attention
		self.MHA = MultiHeadAttention(n_heads = n_heads, embed_dim = embed_dim, need_W = False, is_encoder = False)
		self.SHA = DotProductAttention(clip = clip, return_logits = True, head_depth = embed_dim, is_encoder = False)
		# SHA ==> Single Head Attention, because this layer n_heads = 1 which means no need to spilt heads
		self.env = Env_multicolors if multicolor else Env
		# self.projection_head = nn.Sequential(
        #     nn.Linear(embed_dim,embed_dim),
        #     nn.ReLU(),
        #     nn.Linear(embed_dim,embed_dim)
        # )
		self.multicolor = multicolor

	def compute_static(self, node_embeddings, graph_embedding):
		self.Q_fixed = self.Wq_fixed(graph_embedding[:,None,:]) # (batch, 1, embed_dim) map graph_embedding to Q_fixed
		self.K1 = self.Wk1(node_embeddings) # (batch, n_nodes, embed_dim) map node_embeddings to K1
		self.V = self.Wv(node_embeddings) # map node_embeddings to V
		self.K2 = self.Wk2(node_embeddings) # map node_embeddings to K2
		
	def compute_dynamic(self, mask, step_context, colors = None):
		Q_step = self.Wq_step(step_context) # map step_context to Q_step, 129 -> 128
		Q1 = self.Q_fixed + Q_step
		Q2 = self.MHA([Q1, self.K1, self.V], mask = mask)
		Q2 = self.Wout(Q2) 
		logits = self.SHA([Q2, self.K2, None], colors = colors, mask = mask) 

		return logits.squeeze(dim = 1) # logits: (batch, n_nodes)

	# def compute_color_similarities(self, colors): # used for multi-color cases
		
	# 	batch, n_nodes = colors.shape[:2]
	# 	color_sets = []
	# 	# 计算每个节点的颜色集合
	# 	for i in range (batch):
	# 		this_batch = []
	# 		for j in range(n_nodes):
	# 			color = set()
	# 			color.add(colors[i, j].item())
	# 			this_batch.append(color)
	# 		color_sets.append(this_batch)
	# 	# 计算每对节点之间的颜色相似度

	# 	result = []

	# 	for ba in range(batch):
	# 		this_batch = []
	# 		for i in range(n_nodes):
	# 			this_node = []
	# 			for j in range(n_nodes):
	# 				# 计算颜色集合的交集和并集
	# 				intersection = color_sets[ba][i] & color_sets[ba][j]
	# 				union = color_sets[ba][i] | color_sets[ba][j]
	# 				# 计算相似度
	# 				similarity = len(intersection) / len(union) if len(union) else 0
	# 				this_node.append(similarity)
	# 			this_batch.append(this_node)

	# 		result.append(this_batch)
	# 	# 将相似度转换为张量
	# 	result = torch.tensor(result, dtype=torch.float32)
	# 	return result
	
	# def mask_logits_with_color_similarities(self, logits, colors):
	# 	print("begin to compute color similarities")
	# 	print("colors:", colors.shape)
	# 	print("logits:", logits.shape)
		
	# 	# 计算颜色相似度
	# 	color_similarities = self.compute_color_similarities(colors)
	# 	print("color_similarities:", color_similarities.shape)
		
	# 	# 执行逐元素操作，避免使用循环
	# 	mask1 = (color_similarities > 0) & (logits > 0)	# high similarity, high logits(close distance)
	# 	mask2 = (color_similarities > 0.75) & (logits < 0) # high similarity, low logits(far distance)
	# 	mask3 = (color_similarities == 0)
		
	# 	logits[mask1] *= color_similarities[mask1]
	# 	logits[mask2] *= -0.5
	# 	logits[mask3] = -self.inf
	

	

		
	def forward(self, x, encoder_output, return_pi = False, decode_type = 'sampling'):
		node_embeddings, graph_embedding = encoder_output
		self.compute_static(node_embeddings, graph_embedding) # compute static context
		
		env = Env(x, node_embeddings) if not self.multicolor else Env_multicolors(x, node_embeddings)
		mask, step_context, C = env._create_t1()
		selecter = {'greedy': TopKSampler(), 'sampling': CategoricalSampler()}.get(decode_type, None)
		log_ps, tours = [], []	
		for i in range(env.n_nodes*2): # each nodenhas a different color, so we need to visit depot after one node is visited
			
			# logits = self.compute_dynamic(mask, step_context, x[2]) if self.multicolor \
			# 	else self.compute_dynamic(mask, step_context) 
			logits = self.compute_dynamic(mask, step_context)
    		
      		# here x[2] is the color(set) of each node, multi-color used to mark whether the input data is multi-color
         	
			log_p = torch.log_softmax(logits, dim = -1)
			next_node = selecter(log_p)
			mask, step_context, C = env._get_step(next_node, C)
			tours.append(next_node.squeeze(1)) # remove unnecessary dimension
			log_ps.append(log_p)
			if env.visited_customer.all():
				break

		pi = torch.stack(tours, 1)
		cost = env.get_costs(pi)
		ll = env.get_log_likelihood(torch.stack(log_ps, 1), pi)
		
		if return_pi:
			return cost, ll, pi
		return cost, ll

if __name__ == '__main__':
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	batch, n_nodes, embed_dim = 4, 21, 128
	data = generate_data(device = device, n_samples = batch, n_customer = n_nodes-1)
	decoder = DecoderCell(embed_dim, n_heads = 8, clip = 10.)
	decoder.to(device)
	decoder = torch.nn.DataParallel(decoder)
	node_embeddings = torch.rand((batch, n_nodes, embed_dim), dtype = torch.float, device=device)
	graph_embedding = torch.rand((batch, embed_dim), dtype = torch.float, device=device)
	encoder_output = (node_embeddings, graph_embedding)
	# a = graph_embedding[:,None,:].expand(batch, 7, embed_dim)
	# a = graph_embedding[:,None,:].repeat(1, 7, 1)
	# print(a.size())

	decoder.train()
	cost, ll, pi = decoder(data, encoder_output, return_pi = True, decode_type = 'sampling')
	print('\ncost: ', cost.size(), cost)
	print('\npi: ', pi.size(), pi)
	print('\nll: ', ll.size(), ll)
	

	# cnt = 0
	# for i, k in decoder.state_dict().items():
	# 	print(i, k.size(), torch.numel(k))
	# 	cnt += torch.numel(k)
	# print(cnt)

	# ll.mean().backward()
	# print(decoder.Wk1.weight.grad)
	# https://discuss.pytorch.org/t/model-param-grad-is-none-how-to-debug/52634	