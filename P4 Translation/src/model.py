import torch
from torch import nn
from einops import rearrange


# class Attention(nn.Module):
# 	# TODO: still got work to do
# 	def __init__(self, dim, heads, head_dim, dropout) -> None:
# 		super().__init__()
# 		self.dim=dim
# 		self.heads=heads
# 		self.head_dim=head_dim
# 		inner_dim=heads*head_dim
# 		self.wq=nn.Linear(dim, inner_dim, bias=False)
# 		self.wk=nn.Linear(dim, inner_dim, bias=False)
# 		self.wv=nn.Linear(dim, inner_dim, bias=False)
# 		self.out=nn.Linear(inner_dim, dim)
# 		self.sm=nn.Softmax(dim=-1)
# 		self.drop=nn.Dropout(dropout)
# 		self.norm=nn.LayerNorm(dim)


# 	def forward(self, x):
# 		# input shape: (batch*heads*embed)
# 		x=self.norm(x)
# 		q, k, v=self.wq(x), self.wk(x), self.wv(x)
# 		q, k, v=map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), [q,k,v])
# 		dots=torch.matmul(q, k.transpose(-1,-2))/self.head_dim**0.5
# 		att=self.sm(dots)
# 		att=self.drop(att)
# 		out=torch.matmul(att, v)
# 		out=rearrange(out, 'b h n d -> b n (h d)')
# 		out=self.out(out)
# 		return out


class FeedForward(nn.Module):
	def __init__(self, h_dim, inner_dim, dropout):
		super().__init__()
		self.fc1=nn.Linear(h_dim, inner_dim)
		self.fc2=nn.Linear(inner_dim, h_dim)
		self.dropout=nn.Dropout(dropout)

	def forward(self, x):
		x=torch.relu(self.fc1(x))
		x=self.dropout(x)
		x=self.fc2(x)
		return x


class EncodeLayer(nn.Module):
	def __init__(self, h_dim, n_heads, ff_dim, dropout):
		super().__init__()
		self.attention=nn.MultiheadAttention(h_dim, n_heads, dropout=dropout, batch_first=True)
		self.att_layernorm=nn.LayerNorm(h_dim)
		self.ff=FeedForward(h_dim, ff_dim, dropout)
		self.ff_layernorm=nn.LayerNorm(h_dim)
		
		self.att_dropout=nn.Dropout(dropout)
		self.ff_dropout=nn.Dropout(dropout)

	def forward(self, src, src_pad_mask):
		att_out=self.attention(src, src, src, key_padding_mask=src_pad_mask, need_weights=False)
		att_out=att_out[0]
		out=self.att_layernorm(src+self.att_dropout(att_out))
		ff_out=self.ff(out)
		out=self.ff_layernorm(out+self.ff_dropout(ff_out))
		return out


class Encoder(nn.Module):
	def __init__(self, vocab_size, h_dim, ff_dim, n_heads, n_layers, dropout, max_seq_len):
		super().__init__()
		self.h_dim=h_dim
		self.word_embed=nn.Embedding(vocab_size, h_dim)
		self.pos_embed=nn.Embedding(max_seq_len, h_dim)
		# self.layers=nn.ModuleList([EncodeLayer(h_dim, n_heads, ff_dim, dropout) for _ in range(n_layers)])
		self.layers=nn.ModuleList()
		for _ in range(n_layers):
			self.layers.append(EncodeLayer(h_dim, n_heads, ff_dim, dropout))
		self.dropout=nn.Dropout(dropout)

	def forward(self, src, src_pad_mask):
		scale=torch.sqrt(torch.tensor(self.h_dim))
		out=self.word_embed(src)*scale
		src_bsize, src_len=src.size()[:2]
		pos=torch.arange(0, src_len).unsqueeze(0).repeat(src_bsize, 1).to(src.device)
		out=self.dropout(out+self.pos_embed(pos))
		for layer in self.layers:
			out=layer(out, src_pad_mask)
		return out


class DecodeLayer(nn.Module):
	def __init__(self, h_dim, n_heads, ff_dim, dropout):
		super().__init__()
		self.self_att=nn.MultiheadAttention(h_dim, n_heads, dropout=dropout, batch_first=True)
		self.att=nn.MultiheadAttention(h_dim, n_heads, dropout=dropout, batch_first=True)
		self.ff=FeedForward(h_dim, ff_dim, dropout)

		self.self_att_layernorm=nn.LayerNorm(h_dim)
		self.att_layernorm=nn.LayerNorm(h_dim)
		self.ff_layernorm=nn.LayerNorm(h_dim)

		self.self_att_dropout=nn.Dropout(dropout)
		self.att_dropout=nn.Dropout(dropout)
		self.ff_dropout=nn.Dropout(dropout)

	def forward(self, trg, encoder_out, trg_pad_mask, trg_att_mask):
		self_att_out=self.self_att(trg, trg, trg, key_padding_mask=trg_pad_mask, attn_mask=trg_att_mask, need_weights=False)
		self_att_out=self_att_out[0]
		out=self.self_att_layernorm(trg+self.self_att_dropout(self_att_out))
		att_out=self.att(out, encoder_out, encoder_out, need_weights=False)
		att_out=att_out[0]
		out=self.att_layernorm(out+self.att_dropout(att_out))
		ff_out=self.ff(out)
		out=self.ff_layernorm(out+self.ff_dropout(ff_out))
		return out
		

class Decoder(nn.Module):
	def __init__(self, vocab_size, h_dim, n_heads, ff_dim, n_layers, dropout, max_seq_len):
		super().__init__()
		self.h_dim=h_dim
		self.word_embed=nn.Embedding(vocab_size, h_dim)
		self.pos_embed=nn.Embedding(max_seq_len, h_dim)
		self.layers=nn.ModuleList([DecodeLayer(h_dim, n_heads, ff_dim, dropout) for _ in range(n_layers)])
		self.dropout=nn.Dropout(dropout)
		self.fc=nn.Linear(h_dim, vocab_size)

	def forward(self, trg, encoder_out, trg_pad_mask, trg_att_mask):
		scale=torch.sqrt(torch.tensor(self.h_dim))
		out=self.word_embed(trg)*scale
		trg_bsize, trg_len=trg.size()[:2]
		pos=torch.arange(0, trg_len).unsqueeze(0).repeat(trg_bsize, 1).to(trg.device)
		out=self.dropout(out+self.pos_embed(pos))
		for layer in self.layers:
			out=layer(out, encoder_out, trg_pad_mask, trg_att_mask)
		out=self.fc(out)
		return out


class Bert(nn.Module):
	def __init__(self, src_vocab_size, trg_vocab_size, h_dim, enc_n_heads, dec_n_heads,
			  enc_ff_dim, dec_ff_dim, enc_n_layers, dec_n_layers, dropout, max_src_len, max_trg_len):
		super().__init__()
		self.max_src_len=max_src_len
		self.max_trg_len=max_trg_len
		self.encoder=Encoder(src_vocab_size, h_dim, enc_ff_dim, enc_n_heads, enc_n_layers, dropout, max_src_len)
		self.decoder=Decoder(trg_vocab_size, h_dim, dec_n_heads, dec_ff_dim, dec_n_layers, dropout, max_trg_len)
		

	def forward(self, src, trg, src_pad_mask, trg_pad_mask, trg_att_mask):
		encoder_out=self.encoder(src, src_pad_mask)
		out=self.decoder(trg, encoder_out, trg_pad_mask, trg_att_mask)
		return out




