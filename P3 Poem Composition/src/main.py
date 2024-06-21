import time
import numpy as np
import torch
import torch.utils
import torch.utils.data
from model import Poet
import random
from torch.nn import functional
import setting

LOAD=False
SAVE_FREQ=2
EPOCH=500
BATCH_SIZE=100
LR=0.001

# data loading
data_pack=np.load('../data/tang.npz', allow_pickle=True)
data=torch.from_numpy(data_pack['data'])
ix2word=data_pack['ix2word'].item()
word2ix=data_pack['word2ix'].item()
vocab_size=len(ix2word)
# data preprocessing
def generate_train_data_point(original_data_point: torch.Tensor):
	aug_data=torch.cat((original_data_point, torch.tensor([word2ix['</s>']], dtype=int)))
	x=aug_data[:-1]
	y=aug_data[1:]
	return (x, y)
# train_data=list(map(generate_train_data_point, data))
train_loader=torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

# training
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"we are using {device}")
poet=setting.poet.to(device)
optimizer=torch.optim.Adam(poet.parameters(), lr=LR)
loss_func=torch.nn.CrossEntropyLoss()

epoch=0
train_loss=[]
if LOAD:
	cp=torch.load('../checkpoint/poet.pth')
	poet.load_state_dict(cp['model_state'])
	optimizer.load_state_dict(cp['optimizer_state'])
	epoch=cp['current_epoch']
	train_loss=cp['train_loss']

bnum=len(train_loader)
while epoch<EPOCH:
	start_time=time.time()
	mean_loss=0
	for step, data_point in enumerate(train_loader):
		t_time=time.time()
		xy=list(map(generate_train_data_point, data_point))
		x=torch.stack([e for e, _ in xy])
		y=torch.stack([e for _, e in xy])
		x, y=x.to(device), y.to(device)
		out, _=poet(x)
		loss=loss_func(out.transpose(1, 2), y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		mean_loss+=loss.data.cpu().numpy()
		# print(f"step {step}, loss {loss.data.cpu().numpy()} {time.time()-t_time}")
	epoch+=1
	mean_loss/=bnum
	train_loss.append(mean_loss)
	elapse_time=time.time()-start_time
	print(f"Epoch {epoch}: train_loss= {mean_loss}, using {elapse_time/60:.2f} minutes")
	if epoch%SAVE_FREQ==0:
		checkpoint={
			'model_state': poet.state_dict(),
			'optimizer_state': optimizer.state_dict(),
			'current_epoch': epoch,
			'train_loss': train_loss,
		}
		torch.save(checkpoint, '../checkpoint/poet.pth')

