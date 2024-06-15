import torch
import torchvision
from torch.utils.data import dataloader
from torch import nn
from model import ViT
from torchvision.transforms import v2
from plot import plot_curves

EPOCH=2
BATCH_SIZE=100
LR=0.001

# Data Augmentation
train_transform=v2.Compose([v2.RandomResizedCrop(224),
							v2.RandomVerticalFlip(),
							v2.RandomRotation(degrees=(0,180)),
							v2.ToImage(), 
							v2.ToDtype(torch.float32, scale=True),
							v2.Normalize(mean=[0.485, 0.456, 0.406],
					std=[0.229, 0.224, 0.225])])

test_transform=v2.Compose([v2.Resize(224),
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

vit=ViT(img_size=(224, 224),
		patch_size=(16, 16),
		embed_dim=100,
		transformer_depth=12,
		att_dim=150,
		mlp_dim=300,
		out_dim=10,
		dropout=0.1).to(device)
optimizer=torch.optim.Adam(vit.parameters(), lr=LR)
loss_func=nn.CrossEntropyLoss()

train_loss=[]
test_acc=[]

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

for epoch in range(EPOCH):
	for step, (x, y) in enumerate(train_loader):
		x, y=x.to(device), y.to(device)
		out=vit(x)
		loss=loss_func(out, y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if step%500==499:
			acc=test()
			ls=loss.data.cpu().numpy()
			train_loss.append(ls)
			test_acc.append(acc)
			print(f"Epoch: {epoch+1}| train loss: {ls}| test accuracy: {acc}")

plot_curves(train_loss, test_acc, EPOCH, '../doc/pic/')


		

