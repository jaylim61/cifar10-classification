import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from cnn_model import *

if __name__ == "__main__":
	train_data = torchvision.datasets.CIFAR10(
		root='./data/label_shuffle', train=True, transform=transform, download=True)
	test_data = torchvision.datasets.CIFAR10(
		root='./data/label_shuffle', train=False, transform=transform, download=True)

	np.random.shuffle(train_data.targets)

	train_loader, test_loader = load_data(train_data, test_data, batch_size)

	net = CNN()
	loss_function = nn.CrossEntropyLoss()
	optimizer = optim.Adam(net.parameters(), lr=0.001)

	train(net, train_loader, loss_function, optimizer, epochs=30)

	test(net, test_loader) # Accuracy: 10.0%
	torch.save(net.state_dict(), './trained_models/label_shuffle.pth')
