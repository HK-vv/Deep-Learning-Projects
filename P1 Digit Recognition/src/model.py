from torch import nn

class CNN(nn.Module):
	def __init__(self, *args, **kwargs) -> None:
		super().__init__(*args, **kwargs)
		#->1*1*28
		self.conv1=nn.Conv2d(1, 4, 5, 1, 'same')
		self.conv2=nn.Conv2d(4, 16, 5, 1, 'same')
		self.conv3=nn.Conv2d(16, 64, 5, 1, 'same')
		self.linear=nn.Linear(64*7*7,10)
		self.pool=nn.MaxPool2d(2)
		self.relu=nn.ReLU()
	
	def forward(self, x):
		x=self.conv1(x)
		x=self.relu(x)
		x=self.pool(x)
		x=self.conv2(x)
		x=self.relu(x)
		x=self.pool(x)
		x=self.conv3(x)
		x=self.relu(x)
		x=x.view(x.size(0), -1)
		return self.linear(x)
	


		