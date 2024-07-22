import torch
import torch.nn as nn

class Env():
	def __init__(self, x, node_embeddings):
     
		# node embeddings: (batch, n_nodes, embed_dim) -- embedding for all nodes
		super().__init__()
		""" depot_xy: (batch, 2)
			customer_xy: (batch, n_nodes-1, 2)
			--> self.xy: (batch, n_nodes, 2), Coordinates of depot + customer nodes
			colors: (batch, n_nodes-1)
			node_embeddings: (batch, n_nodes, embed_dim)

			is_next_depot: (batch, 1), e.g., [[True], [True], ...]
			Nodes that have been visited will be marked with True.
		"""
		# self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.depot_xy, customer_xy, self.colors = x
		self.depot_xy, customer_xy, self.colors = self.depot_xy, customer_xy, self.colors
  
		# self.depot_xy: (batch, 2), self.customer_xy: (batch, n_nodes-1, 2), self.xy: (batch, n_nodes, 2) here n_nodes = 21
		self.xy = torch.cat([self.depot_xy[:,None,:], customer_xy], 1)
  
		self.node_embeddings = node_embeddings
		self.batch, self.n_nodes, self.embed_dim = node_embeddings.size()

		self.is_next_depot = torch.ones([self.batch, 1], dtype = torch.bool) 
		self.visited_customer = torch.zeros((self.batch, self.n_nodes-1, 1), dtype = torch.bool)
  
  

	def get_mask_Colors(self, next_node, visited_mask, C):
		""" next_node: ([[0],[0],[not 0], ...], (batch, 1), dtype = torch.int32), [0] denotes going to depot
			visited_mask **includes depot**: (batch, n_nodes, 1)
			visited_mask[:,1:,:] **excludes depot**: (batch, n_nodes-1, 1)
			customer_idx **excludes depot**: (batch, 1), range[0, n_nodes-1] e.g. [[3],[0],[5],[11], ...], [0] denotes 0th customer, not depot
			self.colors **excludes depot**: (batch, n_nodes-1)
			selected_color: (batch, 1)
			if next node is depot, color goes to 0(any node can be chosen next time)
			C: (batch, 1), denotes "current vehicle color"
			self.color_different_customer **excludes depot**: (batch, n_nodes-1)
			visited_customer **excludes depot**: (batch, n_nodes-1, 1)
		 	is_next_depot: (batch, 1), e.g. [[True], [True], ...]
		 	return mask: (batch, n_nodes, 1)		
		"""
		self.is_next_depot = next_node == 0
		# C = C.masked_fill(self.is_next_depot == True, 0) # if next node is depot, color goes to 0(any node can be chosen next time)
		self.visited_customer = self.visited_customer.to(visited_mask.device)
		self.visited_customer = self.visited_customer | visited_mask[:,1:,:]
		customer_idx = torch.argmax(visited_mask[:,1:,:].type(torch.long), dim = 1) # [:,1:,:] begins from 1st customer, not depot, so depot is excluded
		selected_color = torch.gather(input = self.colors, dim = 1, index = customer_idx)
  		#selected_demand = torch.gather(input = self.demand, dim = 1, index = customer_idx)
		# C = torch.zeros([C.size(0),1], dtype = int) if next_node else selected_color
		C = selected_color
		C = C.masked_fill(self.is_next_depot == True, 0) # if next node is depot, color goes to 0(any node can be chosen next time)
		# different_color_customer = torch.zeros(self.colors.size(), dtype = torch.bool) if self.is_next_depot else self.colors != C
		# 初始化 different_color_customer 张量
		
		different_color_customer = torch.ones_like(self.colors, dtype=torch.bool)
		# 按批次循环比较
		for i in range(self.colors.shape[0]):
			# 判断 is_next_depot 是否为 True
			if self.is_next_depot[i][0]:
				# 如果为 True，将 different_color_customer 的该批次置为 False
				different_color_customer[i] = False
			else: # TODO: multiple color case
				# 如果为 False，比较 self.colors 和 C
				equal_mask = torch.eq(self.colors[i], C[i])
				different_color_customer[i] = ~equal_mask

		mask_customer = different_color_customer[:,:,None] | self.visited_customer 
		# mask_customer contains both color-different and visited customer.
  
  
		mask_depot = torch.sum((mask_customer == False).type(torch.long), dim = 1) > 0
		# mask_depot = True when nodes of current colors are not visited fully.(triangle inequality)
		
		return torch.cat([mask_depot[:,None,:], mask_customer], dim = 1), C
	
	def _get_step(self, next_node, C):
		one_hot = torch.eye(self.n_nodes).to(next_node.device)[next_node]
		visited_mask = one_hot.type(torch.bool).permute(0,2,1)

		mask, C = self.get_mask_Colors(next_node, visited_mask, C)
		self.colors = self.colors.masked_fill(self.visited_customer[:,:,0] == True, 10000) # set color of visited customer to 10000
		
		prev_node_embedding = torch.gather(input = self.node_embeddings, dim = 1, index = next_node[:,:,None].repeat(1,1,self.embed_dim))
		# prev_node_embedding = torch.gather(input = self.node_embeddings, dim = 1, index = next_node[:,:,None].expand(self.batch,1,self.embed_dim))

		step_context = torch.cat([prev_node_embedding, C[:,:,None]], dim = -1)
		return mask, step_context, C

	def _create_t1(self): 
		mask_t1 = self.create_mask_t1()
		step_context_t1, C_t1 = self.create_context_C_t1()
		return mask_t1, step_context_t1, C_t1

	def create_mask_t1(self):
		mask_customer = self.visited_customer # visited_customer is initialized as False(no customer is visited)
		mask_depot = torch.ones([self.batch, 1, 1], dtype = torch.bool) # mask_depot is initialized as True)
		return torch.cat([mask_depot, mask_customer], dim = 1) # (batch, n_nodes(1 + n_customers), 1)

	def create_context_C_t1(self):
		C_t1 = torch.zeros([self.batch, 1], dtype=torch.int) # colors is initialized as 0
		depot_idx = torch.zeros([self.batch, 1], dtype = torch.long, device=self.node_embeddings.device) # depot_idx is initialized as 0
		

		depot_embedding = torch.gather(input = self.node_embeddings, dim = 1, index = depot_idx[:,:,None].repeat(1,1,self.embed_dim))
		# depot_embedding = torch.gather(input = self.node_embeddings, dim = 1, index = depot_idx[:,:,None].expand(self.batch,1,self.embed_dim))
		# https://medium.com/analytics-vidhya/understanding-indexing-with-pytorch-gather-33717a84ebc4
		# node_embeddings: (batch, n_nodes, embed_dim) depot_idx: (batch, 1) --> (batch, 1, embed_dim)
		C_t1 = C_t1.to(self.node_embeddings.device)
  
		return torch.cat([depot_embedding, C_t1[:,:, None]], dim = -1), C_t1 # (batch, 1, embed_dim+1), (batch, 1)

	def get_log_likelihood(self, _log_p, pi):
		""" _log_p: (batch, decode_step, n_nodes)
			pi: (batch, decode_step), predicted tour
		"""
		log_p = torch.gather(input = _log_p, dim = 2, index = pi[:,:,None])
		return torch.sum(log_p.squeeze(-1), 1)

	def get_costs(self, pi):
		""" self.xy: (batch, n_nodes, 2), Coordinates of depot + customer nodes
			pi: (batch, decode_step), predicted tour
			d: (batch, decode_step, 2)
			Note: first element of pi is not depot, the first selected node in the path
		"""
		d = torch.gather(input = self.xy, dim = 1, index = pi[:,:,None].repeat(1,1,2))
		# d = torch.gather(input = self.xy, dim = 1, index = pi[:,:,None].expand(self.batch,pi.size(1),2))
		return (torch.sum((d[:, 1:] - d[:, :-1]).norm(p = 2, dim = 2), dim = 1)
				+ (d[:, 0] - self.depot_xy).norm(p = 2, dim = 1)# distance from depot to first selected node
				+ (d[:, -1] - self.depot_xy).norm(p = 2, dim = 1)# distance from last selected node (!=0 for graph with longest path) to depot
				)




class Env_multicolors(): # the only difference is that the colors are multicolors(using and operation)
	def __init__(self, x, node_embeddings):
     
		# node embeddings: (batch, n_nodes, embed_dim) -- embedding for all nodes
		super().__init__()
		""" depot_xy: (batch, 2)
			customer_xy: (batch, n_nodes-1, 2)
			--> self.xy: (batch, n_nodes, 2), Coordinates of depot + customer nodes
			colors: (batch, n_nodes-1)
			node_embeddings: (batch, n_nodes, embed_dim)

			is_next_depot: (batch, 1), e.g., [[True], [True], ...]
			Nodes that have been visited will be marked with True.
		"""
		# self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.depot_xy, customer_xy, self.colors = x
		self.depot_xy, customer_xy, self.colors = self.depot_xy, customer_xy, self.colors
  
		# self.depot_xy: (batch, 2), self.customer_xy: (batch, n_nodes-1, 2), self.xy: (batch, n_nodes, 2) here n_nodes = 21
		self.xy = torch.cat([self.depot_xy[:,None,:], customer_xy], 1)
  
		self.node_embeddings = node_embeddings
		self.batch, self.n_nodes, self.embed_dim = node_embeddings.size()

		self.is_next_depot = torch.ones([self.batch, 1], dtype = torch.bool) 
		self.visited_customer = torch.zeros((self.batch, self.n_nodes-1, 1), dtype = torch.bool)

		# self.compute_single_color_node(self.colors) # compute single color node
  
  

	def get_mask_Colors(self, next_node, visited_mask, C):
		self.is_next_depot = next_node == 0
		# C = C.masked_fill(self.is_next_depot == True, 0) # if next node is depot, color goes to 0(any node can be chosen next time)
		self.visited_customer = self.visited_customer.to(visited_mask.device)
		self.visited_customer = self.visited_customer | visited_mask[:,1:,:]
		customer_idx = torch.argmax(visited_mask[:,1:,:].type(torch.long), dim = 1) # [:,1:,:] begins from 1st customer, not depot, so depot is excluded
		selected_color = torch.gather(input = self.colors, dim = 1, index = customer_idx)
  		#selected_demand = torch.gather(input = self.demand, dim = 1, index = customer_idx)
		# C = torch.zeros([C.size(0),1], dtype = int) if next_node else selected_color
		C = selected_color
		C = C.masked_fill(self.is_next_depot == True, 0) # if next node is depot, color goes to 0(any node can be chosen next time)
		
		# 初始化 different_color_customer 张量
		
		different_color_customer = torch.ones_like(self.colors, dtype=torch.bool)
		# 按批次循环比较
		for i in range(self.colors.shape[0]):
			# 判断 is_next_depot 是否为 True
			if self.is_next_depot[i][0]:
				# 如果为 True，将 different_color_customer 的该批次置为 False
				different_color_customer[i] = False
			else: 
				# 如果为 False，比较 self.colors 和 C (四位二进制数按位或)
				equal_mask = (self.colors[i] & C[i]) > 0
				different_color_customer[i] = ~equal_mask

		mask_customer = different_color_customer[:,:,None] | self.visited_customer 
		# mask_customer contains both color-different and visited customer.
  
  
		# mask_depot = self.is_next_depot & (torch.sum((mask_customer == False).type(torch.long), dim = 1) > 0)
		# this is for cvrp. we cannot choose depot in the continuous time step.
  
  
		mask_depot = torch.sum((mask_customer == False).type(torch.long), dim = 1) > 0
		# mask_depot = True when nodes of current colors are not visited fully.(triangle inequality)
		
		return torch.cat([mask_depot[:,None,:], mask_customer], dim = 1), C
	
	def _get_step(self, next_node, C):
		""" next_node **includes depot** : (batch, 1) int, range[0, n_nodes-1]
			--> one_hot: (batch, 1, n_nodes)
			node_embeddings: (batch, n_nodes, embed_dim)
			demand: (batch, n_nodes-1)
			--> if the customer node is visited, demand goes to 0 
			prev_node_embedding: (batch, 1, embed_dim)
			context: (batch, 1, embed_dim+1)
		"""
		one_hot = torch.eye(self.n_nodes).to(next_node.device)[next_node]
		visited_mask = one_hot.type(torch.bool).permute(0,2,1)

		mask, C = self.get_mask_Colors(next_node, visited_mask, C)
		self.colors = self.colors.masked_fill(self.visited_customer[:,:,0] == True, 10000) # set color of visited customer to 10000
		
		prev_node_embedding = torch.gather(input = self.node_embeddings, dim = 1, index = next_node[:,:,None].repeat(1,1,self.embed_dim))
		# prev_node_embedding = torch.gather(input = self.node_embeddings, dim = 1, index = next_node[:,:,None].expand(self.batch,1,self.embed_dim))

		step_context = torch.cat([prev_node_embedding, C[:,:,None]], dim = -1)
		return mask, step_context, C

	def _create_t1(self): 
		mask_t1 = self.create_mask_t1()
		step_context_t1, C_t1 = self.create_context_C_t1()
		return mask_t1, step_context_t1, C_t1

	def create_mask_t1(self):
		mask_customer = self.visited_customer # visited_customer is initialized as False(no customer is visited)
		mask_depot = torch.ones([self.batch, 1, 1], dtype = torch.bool) # mask_depot is initialized as True)
		return torch.cat([mask_depot, mask_customer], dim = 1) # (batch, n_nodes(1 + n_customers), 1)

	def create_context_C_t1(self):
		C_t1 = torch.zeros([self.batch, 1], dtype=torch.int) # colors is initialized as 0
		depot_idx = torch.zeros([self.batch, 1], dtype = torch.long, device=self.node_embeddings.device) # depot_idx is initialized as 0
		

		depot_embedding = torch.gather(input = self.node_embeddings, dim = 1, index = depot_idx[:,:,None].repeat(1,1,self.embed_dim))
		# depot_embedding = torch.gather(input = self.node_embeddings, dim = 1, index = depot_idx[:,:,None].expand(self.batch,1,self.embed_dim))
		# https://medium.com/analytics-vidhya/understanding-indexing-with-pytorch-gather-33717a84ebc4
		# node_embeddings: (batch, n_nodes, embed_dim) depot_idx: (batch, 1) --> (batch, 1, embed_dim)
		C_t1 = C_t1.to(self.node_embeddings.device)
  
		return torch.cat([depot_embedding, C_t1[:,:, None]], dim = -1), C_t1 # (batch, 1, embed_dim+1), (batch, 1)

	def get_log_likelihood(self, _log_p, pi):
		""" _log_p: (batch, decode_step, n_nodes)
			pi: (batch, decode_step), predicted tour
		"""
		log_p = torch.gather(input = _log_p, dim = 2, index = pi[:,:,None])
		return torch.sum(log_p.squeeze(-1), 1)

	def get_costs(self, pi):
		""" self.xy: (batch, n_nodes, 2), Coordinates of depot + customer nodes
			pi: (batch, decode_step), predicted tour
			d: (batch, decode_step, 2)
			Note: first element of pi is not depot, the first selected node in the path
		"""
		d = torch.gather(input = self.xy, dim = 1, index = pi[:,:,None].repeat(1,1,2))
		# d = torch.gather(input = self.xy, dim = 1, index = pi[:,:,None].expand(self.batch,pi.size(1),2))
		return (torch.sum((d[:, 1:] - d[:, :-1]).norm(p = 2, dim = 2), dim = 1)
				+ (d[:, 0] - self.depot_xy).norm(p = 2, dim = 1)# distance from depot to first selected node
				+ (d[:, -1] - self.depot_xy).norm(p = 2, dim = 1)# distance from last selected node (!=0 for graph with longest path) to depot
				)




class Sampler(nn.Module):
	""" args; logits: (batch, n_nodes)
		return; next_node: (batch, 1)
		TopKSampler <=> greedy; sample one with biggest probability
		CategoricalSampler <=> sampling; randomly sample one from possible distribution based on probability
	"""
	def __init__(self, n_samples = 1, **kwargs):
		super().__init__(**kwargs)
		self.n_samples = n_samples
		
class TopKSampler(Sampler): # greedy
	def forward(self, logits):
		return torch.topk(logits, self.n_samples, dim = 1)[1]# == torch.argmax(log_p, dim = 1).unsqueeze(-1)
		# torch.topk returns a tuple of two tensors: values and indices. by chosing [1], we get indices of topk nodes

class CategoricalSampler(Sampler): # sampling
	def forward(self, logits):
		return torch.multinomial(logits.exp(), self.n_samples) # n_samples = 1
