import math
import numpy as np
import os

import torch
from torchvision import datasets, transforms


class TwoCropTransform :
	# cerate two crops of the same image
	def __init__(self, transform):
		self.transform = transform

	def __call__(self, x):
		return [self.transform(x), self.transform(x)]


def custom_data_loader(batch_size):

	# for cifar10
	mean = (0.4914, 0.4822, 0.4465)
	std = (0.2023, 0.1994, 0.2010)

	train_transform = transforms.Compose([
		transforms.RandomResizedCrop(size = 32, scale = (0.2, 1)),
		transforms.RandomHorizontalFlip(),
		transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p =0.8),
		transforms.RandomGrayscale(p=0.2),
		transforms.ToTensor(),
		transforms.Normalize(mean = mean, std = std)
		])

	if not os.path.isdir('data'):
		os.mkdir('data')

	train_dataset = datasets.CIFAR10('data', transform = TwoCropTransform(train_transform), download = True)
	# train_dataset shape : num_images * ([image, aug_image], label)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle = True, drop_last = True)
	return train_loader


def train(train_loader, model, criterion, optimizer, device, epochs):

	model.train()
	train_loss_list = []
	for epoch in range(epochs):
		epoch_loss = 0
		for idx, (images, labels) in enumerate(train_loader):

			# 512 dimensional features only if ResNet18 is used

			# images = torch.cat((images[0], images[1]), dim = 0)
			# # images[0] : batch_size x 3 x 32 x 32
			# # images[1] : batch_size x 3 x 32 x 32
			# # after concatenation images : 2*batch_size x 32 x 32

			images = torch.cat(images)
			labels = labels.repeat(2)
			images = images.to(device)
			labels = labels.to(device)

			projections = model(images)
			projections = projections.to(device)
			loss = criterion(projections, labels)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			loss = loss.item()
			epoch_loss += loss
			running_average_loss = epoch_loss / (idx + 1)

			print("batch : " + str(idx) + " / 195, " + " epoch : " + str(epoch))
			print("batch loss " + str(loss))
			print("running average loss : " + str(running_average_loss))











		



















