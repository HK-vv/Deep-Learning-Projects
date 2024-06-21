from model import Poet
import numpy as np
import torch

data_pack=np.load('../data/tang.npz', allow_pickle=True)
data=torch.from_numpy(data_pack['data'])
ix2word=data_pack['ix2word'].item()
word2ix=data_pack['word2ix'].item()
vocab_size=len(ix2word)

embed_dim=100
hidden_dim=1000
num_layer=2
mlp_dim=200
poet=Poet(embed_dim=embed_dim, 
		  hidden_dim=hidden_dim, 
		  num_layer=num_layer, 
		  mlp_dim=mlp_dim, 
		  vocab_size=vocab_size)
cp=torch.load('../checkpoint/poet.pth')