import torch
import torchvision
from torch.utils.data import dataloader
from torch import nn
from model import ViT
from torchvision.transforms import v2
from plot import plot_curves
import time

LOAD=False
SAVE_FREQ=5
EPOCH=500
BATCH_SIZE=500
LR=0.001
AUG_SIZE=32*4

# Data Augmentation
train_transform=v2.Compose([v2.RandomResizedCrop(AUG_SIZE),
							v2.RandomVerticalFlip(),
							v2.RandomRotation(degrees=(0,180)),
							v2.ToImage(), 
							v2.ToDtype(torch.float32, scale=True),
							v2.Normalize(mean=[0.485, 0.456, 0.406],
					std=[0.229, 0.224, 0.225])])

test_transform=v2.Compose([v2.Resize(AUG_SIZE),
						   v2.ToImage(), 
						   v2.ToDtype(torch.float32, scale=True),
						   v2.Normalize(mean=[0.485, 0.456, 0.406],
					std=[0.229, 0.224, 0.225])])

# Data Preparation
train_set=torchvision.datasets.CIFAR10(download=False, root='../data', train=True, 
									   transform=train_transform)

test_set=torchvision.datasets.CIFAR10(download=False, root='../data', train=False, 
									  transform=test_transform)

train_loader=dataloader.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader=dataloader.DataLoader(test_set, batch_size=BATCH_SIZE, )

idx2cls=torchvision.datasets.CIFAR10(root='../data').classes
cls2idx=torchvision.datasets.CIFAR10(root='../data').class_to_idx

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"we are using {device}")

vit=ViT(img_size=(AUG_SIZE, AUG_SIZE),
		patch_size=(16, 16),
		embed_dim=300,
		transformer_depth=12,
		att_dim=300,
		mlp_dim=1500,
		out_dim=10,
		dropout=0.1).to(device)
optimizer=torch.optim.Adam(vit.parameters(), lr=LR)
loss_func=nn.CrossEntropyLoss()
epoch=0
train_loss=[]
test_accuracy=[]

if LOAD:
	cp=torch.load('../checkpoint/vit.pth')
	vit.load_state_dict(cp['model_state'])
	optimizer.load_state_dict(cp['optimizer_state'])
	epoch=cp['current_epoch']
	train_loss=cp['train_loss']
	test_accuracy=cp['test_accuracy']


def test():
	correct=total=0.0
	with torch.no_grad():
		for _, (tx, ty) in enumerate(test_loader):
			tx=tx.to(device)
			out=vit(tx)
			pred=torch.argmax(out, dim=1).detach().cpu().numpy()
			correct+=(pred==ty.detach().numpy()).astype(int).sum()
			total+=ty.size(0)
		acc=correct/total
		return acc


bnum=len(train_loader)
while epoch<EPOCH:
	start_time=time.time()
	mean_loss=0
	for step, (x, y) in enumerate(train_loader):
		x, y=x.to(device), y.to(device)
		out=vit(x)
		loss=loss_func(out, y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		mean_loss+=loss.data.cpu().numpy()
	epoch+=1
	acc=test()
	mean_loss/=bnum
	train_loss.append(mean_loss)
	test_accuracy.append(acc)
	elapse_time=time.time()-start_time
	print(f"Epoch {epoch}: train_loss= {mean_loss}, test_accuracy= {acc}, using {elapse_time/60:.2f} minutes")
	if epoch%SAVE_FREQ==0:
		checkpoint={
			'model_state': vit.state_dict(),
			'optimizer_state': optimizer.state_dict(),
			'current_epoch': epoch,
			'train_loss': train_loss,
			'test_accuracy': test_accuracy,
		}
		torch.save(checkpoint, '../checkpoint/vit.pth')

plot_curves(train_loss, test_accuracy, EPOCH, '../doc/pic/')


		

