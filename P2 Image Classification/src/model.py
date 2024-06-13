import torch
from torch import nn

nn.Transformer()

class Attention(nn.Module):
	def __init__(self, dim, heads, head_dim) -> None:
		super().__init__()
		self.dim=dim
		self.heads=heads
		self.head_dim=head_dim
		self.wq=nn.Linear(dim, head_dim, bias=False)
		self.wk=nn.Linear(dim, head_dim, bias=False)
		self.wv=nn.Linear(dim, head_dim, bias=False)
		self.out=nn.Linear(head_dim, dim)


	def forward(self, x):
		# input shape: (batch*heads*embed)
		q=self.wq(x)
		k=self.wk(x)
		v=self.wv(x)
		dots=torch.matmul(q, k.transpose(-1,-2))/self.head_dim**0.5
		att=nn.Softmax(dim=-1)
		out=torch.matmul(att, v)
		out=self.out(out)
		return out
	

class FeedForward(nn.Module):
	def __init__(self, dim) -> None:
		super().__init__()
		self.dim=dim
		self.lin1=nn.Linear(dim, dim)
		self.lin2=nn.Linear(dim, dim)
		self.relu=nn.ReLU()

	def forward(self, x):
		x=self.relu(self.lin1(x))
		return self.lin2(x)
	

class Transformer(nn.Module):
	def __init__(self, N, dim, heads, att_dim) -> None:
		super().__init__()
		self.norm=nn.LayerNorm(dim)
		self.layers=nn.ModuleList([])
		for _ in range(N):
			self.layers.append(nn.ModuleList([Attention(dim, heads, att_dim), 
									 FeedForward(dim)]))
		


	def forward(self, x):
		for att, ff in range(N):
			x=self.norm(torch.add(x, att(x)))
			x=self.norm(torch.add(x, ff(x)))
		return x
	

class MLP(nn.Module):
	def __init__(self, in_dim, out_dim) -> None:
		super().__init__()
		self.lin1=nn.Linear(in_dim, in_dim)
		self.lin2=nn.Linear(in_dim, out_dim)
		self.act=nn.ReLU()

	def forward(self, x):
		x=self.act(self.lin1(x))
		x=self.lin2(x)
		return x


class ViT(nn.Module):
	def __init__(self, ) -> None:
		super().__init__()


	def forward(self, ):
		pass