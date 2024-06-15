import matplotlib.pyplot as plt
import numpy as np

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

