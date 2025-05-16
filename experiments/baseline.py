import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from cnn_model import *

if __name__ == "__main__":
	train_data = torchvision.datasets.CIFAR10(
		root='./data/baseline', train=True, transform=transform, download=True)
	test_data = torchvision.datasets.CIFAR10(
		root='./data/baseline', train=False, transform=transform, download=True)

	train_loader, test_loader = load_data(train_data, test_data, batch_size)

	net = CNN()
	loss_function = nn.CrossEntropyLoss()
	optimizer = optim.Adam(net.parameters(), lr=0.001)

	train(net, train_loader, loss_function, optimizer, epochs=30)

	test(net, test_loader) # Accuracy: 78.61%
	torch.save(net.state_dict(), './trained_models/baseline.pth')
