import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from cnn_model import *

perturbation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.2, hue=0.2),

    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=True),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

if __name__ == "__main__":
	train_data = torchvision.datasets.CIFAR10(
		root='./data/input_perturbation', train=True, transform=perturbation, download=True)
	test_data = torchvision.datasets.CIFAR10(
		root='./data/input_perturbation', train=False, transform=transform, download=True)

	train_loader, test_loader = load_data(train_data, test_data, batch_size)

	net = CNN()
	loss_function = nn.CrossEntropyLoss()
	optimizer = optim.Adam(net.parameters(), lr=0.001)

	train(net, train_loader, loss_function, optimizer, epochs=30)

	test(net, test_loader) # Accuracy: 59.85%
	torch.save(net.state_dict(), './trained_models/input_perturbation.pth')
