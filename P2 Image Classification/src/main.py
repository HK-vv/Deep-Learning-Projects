import torch
import torchvision
from torch.utils.data import dataloader
from torch import nn
from model import ViT
from torchvision.transforms import v2
from plot import plot_curves
import time

LOAD=True
SAVE_FREQ=2
EPOCH=500
BATCH_SIZE=256
LR=0.001
AUG_SIZE=32
PATCH_SIZE=4

# Data Augmentation
train_transform=v2.Compose([v2.RandomVerticalFlip(),
                            v2.RandomHorizontalFlip(),
							v2.RandomRotation(degrees=(0,180)),
							v2.RandomResizedCrop(AUG_SIZE),
							v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
							v2.ToImage(), 
							v2.ToDtype(torch.float32, scale=True),
							v2.Normalize(mean=[0.485, 0.456, 0.406],
							std=[0.229, 0.224, 0.225])])

cifar_transform=v2.Compose([v2.Resize(AUG_SIZE),
							v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10),
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
train_set=torchvision.datasets.CIFAR10(download=True, root='../data', train=True, 
									   transform=cifar_transform)

test_set=torchvision.datasets.CIFAR10(download=False, root='../data', train=False, 
									  transform=test_transform)

train_loader=dataloader.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader=dataloader.DataLoader(test_set, batch_size=BATCH_SIZE, )

idx2cls=torchvision.datasets.CIFAR10(root='../data').classes
cls2idx=torchvision.datasets.CIFAR10(root='../data').class_to_idx

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"we are using {device}")

vit=ViT(img_size=(AUG_SIZE, AUG_SIZE),
		patch_size=(PATCH_SIZE, PATCH_SIZE),
		embed_dim=256,
		transformer_depth=8,
		heads=8,
		mlp_dim=2048,
		out_dim=10,
		dropout=0.2).to(device)
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
	vit.train()
	for step, (x, y) in enumerate(train_loader):
		x, y=x.to(device), y.to(device)
		out=vit(x)
		loss=loss_func(out, y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		mean_loss+=loss.data.cpu().numpy()
	epoch+=1
	vit.eval()
	acc=test()
	mean_loss/=bnum
	train_loss.append(mean_loss)
	test_accuracy.append(acc)
	elapse_time=time.time()-start_time
	print(f"Epoch {epoch}: train_loss={mean_loss:.6f}, test_accuracy={acc*100:.2f}%, using {elapse_time/60:.2f} minutes")
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


		

