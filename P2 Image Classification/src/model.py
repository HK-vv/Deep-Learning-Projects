import torch
from torch import nn
from einops.layers.torch import Rearrange
from einops import repeat

class Attention(nn.Module):
	def __init__(self, dim, heads, head_dim, dropout) -> None:
		super().__init__()
		self.dim=dim
		self.heads=heads
		self.head_dim=head_dim
		self.wq=nn.Linear(dim, head_dim, bias=False)
		self.wk=nn.Linear(dim, head_dim, bias=False)
		self.wv=nn.Linear(dim, head_dim, bias=False)
		self.out=nn.Linear(head_dim, dim)
		self.sm=nn.Softmax(dim=-1)
		self.drop=nn.Dropout(dropout)
		self.norm=nn.LayerNorm(dim)


	def forward(self, x):
		# input shape: (batch*heads*embed)
		x=self.norm(x)
		q=self.wq(x)
		k=self.wk(x)
		v=self.wv(x)
		dots=torch.matmul(q, k.transpose(-1,-2))/self.head_dim**0.5
		att=self.sm(dots)
		att=self.drop(att)
		out=torch.matmul(att, v)
		out=self.out(out)
		return out
	

class FeedForward(nn.Module):
	def __init__(self, dim, dropout) -> None:
		super().__init__()
		self.dim=dim
		self.lin1=nn.Linear(dim, dim)
		self.lin2=nn.Linear(dim, dim)
		self.relu=nn.ReLU()
		self.drop=nn.Dropout(dropout)
		self.norm=nn.LayerNorm(dim)

	def forward(self, x):
		x=self.norm(x)
		x=self.relu(self.lin1(x))
		x=self.drop(x)
		x=self.lin2(x)
		x=self.drop(x)
		return x
	

class Transformer(nn.Module):
	def __init__(self, depth, dim, heads, att_dim, dropout) -> None:
		super().__init__()
		self.N=depth
		self.dim=dim
		self.heads=heads
		self.norm=nn.LayerNorm(dim)
		self.layers=nn.ModuleList([])
		for _ in range(self.N):
			self.layers.append(nn.ModuleList([Attention(dim, heads, att_dim, dropout=dropout), 
									 FeedForward(dim, dropout=dropout)]))
		


	def forward(self, x):
		for att, ff in self.layers:
			x=x+self.norm(att(x))
			x=x+self.norm(ff(x))
		return x
	

class MLP(nn.Module):
	def __init__(self, in_dim, inner_dim, out_dim, dropout) -> None:
		super().__init__()
		self.lin1=nn.Linear(in_dim, inner_dim)
		self.lin2=nn.Linear(inner_dim, out_dim)
		self.act=nn.ReLU()
		self.drop=nn.Dropout(dropout)
		self.norm=nn.LayerNorm(in_dim)

	def forward(self, x):
		x=self.norm(x)
		x=self.act(self.lin1(x))
		x=self.drop(x)
		x=self.lin2(x)
		x=self.drop(x)
		return x


class ImageEmbedding(nn.Module):
	def __init__(self, img_size, patch_size, embed_dim, color) -> None:
		super().__init__()
		self.img_h, self.img_w=img_size
		self.pch_h, self.pch_w=patch_size
		assert self.img_h%self.pch_h==0 and self.img_w%self.pch_w==0
		self.heads=(self.img_h//self.pch_h)*(self.img_w//self.pch_w)
		self.embed_patch=nn.Sequential(
			nn.Conv2d(color, embed_dim, kernel_size=patch_size, stride=patch_size),
			Rearrange('b e h w -> b (h w) e'),
			nn.LayerNorm(embed_dim)
		)
		self.cls_token=nn.Parameter(torch.randn(1, 1, embed_dim))
		self.positions=nn.Parameter(torch.randn(1, self.heads+1, embed_dim))

	def forward(self, x):
		x=self.embed_patch(x)
		# add cls_token
		b, _, _=x.shape
		cls_tokens=repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
		x=torch.cat((cls_tokens, x), dim=1)
		x=x+self.positions
		return x


class ViT(nn.Module):
	def __init__(self, img_size, patch_size, 
			  embed_dim, transformer_depth, att_dim, mlp_dim, out_dim, color=3, dropout=0.0):
		super().__init__()
		self.heads=sum(torch.tensor(img_size)//torch.tensor(patch_size))
		# TODO: embed image patches to vec
		self.img_imbedding=ImageEmbedding(img_size=img_size,
									patch_size=patch_size,
									embed_dim=embed_dim,
									color=color)
		self.transformer=Transformer(depth=transformer_depth, 
							   dim=embed_dim, 
							   heads=self.heads,
							   att_dim=att_dim,
							   dropout=dropout)
		self.mlp=MLP(embed_dim, mlp_dim, out_dim, dropout=dropout)
		self.drop=nn.Dropout(dropout)


	def forward(self, x):
		x=self.img_imbedding(x)
		x=self.drop(x)
		x=self.transformer(x)
		x=x[:,0]
		x=self.mlp(x)
		return x

