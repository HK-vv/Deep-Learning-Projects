import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_curves(train_loss, test_acc, epoch, root):
	x_axis=np.linspace(1, epoch, epoch)
	plt.title('Training Loss')
	plt.plot(x_axis, train_loss)
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.savefig(root+'training_loss.png')

	plt.clf()
	plt.title('Test Accuracy')
	plt.plot(x_axis, test_acc)
	plt.xlabel('epoch')
	plt.ylabel('accuracy')
	plt.savefig(root+'test_accuracy.png')

def plot_checkpoint():
	cp=torch.load('../checkpoint/vit.pth')
	epoch=cp['current_epoch']
	train_loss=cp['train_loss']
	test_accuracy=cp['test_accuracy']
	plot_curves(train_loss, test_accuracy, epoch, root='../doc/pic/')
	print(f"plotted, epoch={epoch}")

if __name__=='__main__':
	plot_checkpoint()