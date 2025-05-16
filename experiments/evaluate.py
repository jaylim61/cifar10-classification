import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision

import seaborn as sns
from sklearn.metrics import confusion_matrix
from cnn_model import *

# Evaluate models and create confusion matrix
def create_confusion_matrix(net, test_loader, classes, save_path=None, title='Confusion Matrix'):
	net.eval()
	all_preds = []
	all_labels = []

	with torch.no_grad():
		for data in test_loader:
			images, labels = data
			outputs = net(images)
			_, predicted = torch.max(outputs, 1)
			all_preds.extend(predicted.cpu().numpy())
			all_labels.extend(labels.cpu().numpy())

	cm = confusion_matrix(all_labels, all_preds)
	plt.figure(figsize=(10, 8))
	sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
	plt.xlabel("Predicted")
	plt.ylabel("Actual")
	plt.title(title)

	if save_path:
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
	plt.show()

if __name__ == "__main__":
	test_data = torchvision.datasets.CIFAR10(
		root='./data/test', train=False, transform=transform, download=True)

	test_loader = torch.utils.data.DataLoader(
		test_data, batch_size=batch_size, shuffle=True, num_workers=2)
	
	# Baseline
	net_baseline = CNN()
	net_baseline.load_state_dict(torch.load('./trained_models/baseline.pth'))
	create_confusion_matrix(
		net_baseline, test_loader, classes, save_path='../report/figures/cm_baseline.png', title='Baseline')
	
	# Random label shuffle
	net_random_shuffle = CNN()
	net_random_shuffle.load_state_dict(torch.load('./trained_models/label_shuffle.pth'))
	create_confusion_matrix(
		net_random_shuffle, test_loader, classes, save_path='../report/figures/cm_label_shuffle.png', title='Random Label Shuffle')

	# Label noise
	net_label_noise = CNN()
	net_label_noise.load_state_dict(torch.load('./trained_models/label_noise.pth'))
	create_confusion_matrix(
		net_label_noise, test_loader, classes, save_path='../report/figures/cm_label_noise.png', title='Label Noise (20%)')

	# Input perturbation
	net_input_perturbation = CNN()
	net_input_perturbation.load_state_dict(torch.load('./trained_models/input_perturbation.pth'))
	create_confusion_matrix(
		net_input_perturbation, test_loader, classes, save_path='../report/figures/cm_input_perturbation.png', title='Input Perturbation')
