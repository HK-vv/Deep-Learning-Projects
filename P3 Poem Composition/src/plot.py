import matplotlib.pyplot as plt
import numpy as np
import torch
import setting

def plot_curves(train_loss, epoch, root):
	x_axis=np.linspace(1, epoch, epoch)
	plt.title('Training Loss')
	plt.plot(x_axis, train_loss)
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.savefig(root+'training_loss.png')


def plot_checkpoint():
	cp=setting.cp
	epoch=cp['current_epoch']
	train_loss=cp['train_loss']
	plot_curves(train_loss, epoch, root='../doc/pic/')

if __name__=='__main__':
	plot_checkpoint()