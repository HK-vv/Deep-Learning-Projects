from torch import nn
import torch


class Embedding(nn.Module):
	def __init__(self, num_embedding, embed_dim) -> None:
		super().__init__()
		self.tune_embedding=nn.Embedding(num_embedding, embed_dim)

	def forward(self, x):
		return self.tune_embedding(x)
	
class MLP(nn.Module):
	def __init__(self, in_dim, mid_dim, out_dim) -> None:
		super().__init__()
		self.lin1=nn.Linear(in_dim, mid_dim)
		self.ln=nn.LayerNorm(mid_dim)
		self.lin2=nn.Linear(mid_dim, out_dim)
		self.act=nn.ReLU()

	def forward(self, x):
		x=self.act(self.lin1(x))
		x=self.ln(x)
		x=self.act(self.lin2(x))
		return x


# word embedding
class Poet(nn.Module):
	def __init__(self, embed_dim, hidden_dim, num_layer, mlp_dim, vocab_size) -> None:
		super().__init__()
		self.embed_dim=embed_dim
		self.hidden_dim=hidden_dim
		self.num_layer=num_layer
		self.word_embed=Embedding(vocab_size, embed_dim=embed_dim)
		self.gru=nn.GRU(embed_dim, hidden_dim, num_layers=num_layer, batch_first=True)
		self.mlp=MLP(hidden_dim, mlp_dim, vocab_size)
		self.sm=nn.Softmax(dim=-1)

	def forward(self, x, state=None):
		if len(x.size())==1:
			batch_num, seq_len=None, x.size()[0]
		else:
			batch_num, seq_len=x.size()
		x=self.word_embed(x)
		if state is None:
			h_0=self.init_state(batch_num).to(x.device)
		else:
			h_0=state
		x, h=self.gru(x, h_0)
		x=self.mlp(x)
		return x, h
	
	def init_state(self, batch_size=None):
		if batch_size is None:
			return torch.zeros(self.num_layer, self.hidden_dim)
		else:
			return torch.zeros(self.num_layer, batch_size, self.hidden_dim)



