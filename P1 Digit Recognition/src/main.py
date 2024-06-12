from random import shuffle
from turtle import backward
import torch
import torchvision
from torch import nn
from torch.utils.data import dataloader
from model import CNN
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE=100
EPOCH=10
LR=0.0005 # learning rate

train_data=torchvision.datasets.MNIST(
	download=False,
	root='../data/',
	train=True,
	transform=torchvision.transforms.ToTensor()
)

test_data=torchvision.datasets.MNIST(
	download=False,
	root='../data/',
	train=False,
	transform=torchvision.transforms.ToTensor()
)

train_loader=dataloader.DataLoader(
	dataset=train_data,
	batch_size=BATCH_SIZE,
	shuffle=True
)

test_loader=dataloader.DataLoader(
	dataset=test_data,
	batch_size=BATCH_SIZE,
)

cnn=CNN()
optimizer=torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func=nn.CrossEntropyLoss()

# Training
train_lost=[]
test_acc=[]
for epoch in range(EPOCH):
	for step, (x,y) in enumerate(train_loader):
		out=cnn(x)
		loss=loss_func(out, y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if step%100==99:
			correct=total=0.0
			with torch.no_grad():
				for i, (tx, ty) in enumerate(test_loader):
					out=cnn(tx)
					# print(out)
					pred=torch.argmax(out, 1).detach().numpy()
					correct+=(pred==ty.detach().numpy()).astype(int).sum()
					total+=float(ty.size(0))
				acc=correct/total
			train_lost.append(loss.data.numpy())
			test_acc.append(acc)
			print(f"Epoch: {epoch+1}| train loss: {loss.data.numpy()}| test accuracy: {acc}")

# Plotting
x_axis=torch.linspace(0, EPOCH, EPOCH*6).numpy()
plt.title('Training Loss')
plt.plot(x_axis, train_lost)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('../doc/pic/training_loss.png')

plt.clf()
plt.title('Test Accuracy')
plt.plot(x_axis, test_acc)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.savefig('../doc/pic/test_accuracy.png')


