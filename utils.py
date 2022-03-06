import math
import numpy as np
import os
import matplotlib.pyplot as plt


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


def train_SupCon_model(train_loader, model, criterion, optimizer, device, epochs):

	model.train()
	train_loss_list = []
	for epoch in range(1,epochs+1):
		running_average_loss = 0
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
			print(" epoch : " + str(epoch) + ", batch : " + str(idx) + " / 194, " + " bl " + str(loss) + ", ral : " + str(running_average_loss))

		train_loss_list.append(epoch_loss)
		print("Epoch " + str(epoch) + "loss : " + str(epoch_loss))
		plot_SupCon_loss(epoch,  train_loss_list)


def train_cross_entropy_model(data_loader, model,  criterion, optimizer, device, epochs):
	
	model.train()
	train_loss_list = []
	train_acc_list = []
	for epoch in range(1,epochs+1):
		running_average_loss = 0
		epoch_loss = 0
		epoch_correct = 0
		total_predictions = 0

		for idx, (images, labels) in enumerate(data_loader):
			images = torch.cat(images)
			labels = labels.repeat(2)
			images = images.to(device)
			labels = labels.to(device)
			optimizer.zero_grad()
			outputs = model(images)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			epoch_loss += loss.item()
			_, predicted = outputs.max(1)
			batch_size = labels.size(0)
			total_predictions += batch_size
			correctly_predicted = predicted.eq(labels).sum().item()
			epoch_correct += correctly_predicted
			batch_acc = correctly_predicted / batch_size

			print("epoch : " + str(epoch) + " , batch : "  +  str(idx) + " / 194, batch_acc : " + str(batch_acc))

		epoch_acc = epoch_correct / total_predictions
		train_acc_list.append(epoch_acc)
		train_loss_list.append(epoch_loss)
		print("Epoch " + str(epoch) + " acc : " + str(epoch_acc))
		print("Epoch " + str(epoch) + " loss : " + str(epoch_loss))
		plot_CCE_loss(epoch, train_loss_list)
		plot_CCE_acc(epoch, train_acc_list)






def plot_SupCon_loss(epochs, losses):

	if not os.path.isdir('Plots'):
		os.mkdir('Plots')
	plt.plot(range(epochs), losses)
	plt.xlabel('Epochs')
	plt.ylabel('Supervised Contrastive Loss')
	plt.savefig('Plots/SupConLoss.jpeg')


def plot_CCE_loss(epochs, losses):

	if not os.path.isdir('Plots'):
		os.mkdir('Plots')
	plt.plot(range(epochs), losses)
	plt.xlabel('Epochs')
	plt.ylabel('CCE Loss')
	plt.savefig('Plots/CCE_Loss.jpeg')

def plot_CCE_acc(epochs, losses):

	if not os.path.isdir('Plots'):
		os.mkdir('Plots')
	plt.plot(range(epochs), losses)
	plt.xlabel('Epochs')
	plt.ylabel('CCE Acc')
	plt.savefig('Plots/CCE_Acc.jpeg')
			



















