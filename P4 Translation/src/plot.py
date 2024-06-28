import matplotlib.pyplot as plt
import numpy as np
import torch
from setting import *

def plot_curves(train_loss, valid_loss, valid_bleu, test_bleu, epoch):
	# x_axis=np.linspace(1, epoch, epoch)
	x_axis=np.arange(1, epoch+1)
	plt.title('Training Loss')
	plt.plot(x_axis, train_loss)
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.savefig(FileName.train_loss_fig)

	plt.clf()
	plt.title('Validation Loss')
	plt.plot(x_axis, valid_loss)
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.savefig(FileName.valid_loss_fig)

	plt.clf()
	plt.title('Validation Bleu Score')
	plt.plot(x_axis, valid_bleu)
	plt.xlabel('epoch')
	plt.ylabel('bleu score')
	plt.savefig(FileName.valid_bleu_fig)

	plt.clf()
	x_axis=np.arange(TEST_FREQ, epoch+1, TEST_FREQ)
	plt.title('Test result')
	plt.plot(x_axis, test_bleu)
	plt.xlabel('epoch')
	plt.ylabel('bleu score')
	plt.savefig(FileName.test_bleu_fig)

def plot_checkpoint():
	cp=torch.load(FileName.checkpoint)
	epoch=cp['current_epoch']
	train_loss=cp['train_loss']
	valid_loss=cp['valid_loss']
	valid_bleu=cp['valid_bleu']
	test_bleu=cp['test_bleu']
	plot_curves(train_loss, valid_loss, valid_bleu, test_bleu, epoch)
	print(f"plotted, epoch={epoch}")

if __name__=='__main__':
	plot_checkpoint()