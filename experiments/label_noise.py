import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from cnn_model import *

if __name__ == "__main__":
	train_data = torchvision.datasets.CIFAR10(
		root='./data/label_noise', train=True, transform=transform, download=True)
	test_data = torchvision.datasets.CIFAR10(
		root='./data/label_noise', train=False, transform=transform, download=True)

	noise_ratio = 0.2
	size = len(train_data.targets)
	noise_index = set(np.random.choice(size, int(size * noise_ratio), replace=False))

	for i in range(size):
		if i in noise_index:
			train_data.targets[i] = int(np.random.choice([label for label in range(10) if label != train_data.targets[i]]))

	train_loader, test_loader = load_data(train_data, test_data, batch_size)

	net = CNN()
	loss_function = nn.CrossEntropyLoss()
	optimizer = optim.Adam(net.parameters(), lr=0.001)

	train(net, train_loader, loss_function, optimizer, epochs=30)

	test(net, test_loader) # Accuracy: 73.41%
	torch.save(net.state_dict(), './trained_models/label_noise.pth')
