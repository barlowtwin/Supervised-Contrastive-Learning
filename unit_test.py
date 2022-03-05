import torch
from resnet import resnet18, resnet34
from SupCon import SupConProjection
import torchvision
import os
from utils import data_loader

# # testing for resnet 

# model = resnet18(in_channels = 3)
# inp = torch.rand(10,3,32,32)
# out = model(inp)
# print(out.size())

# #testing for resnet34 model

# model = resnet34(in_channels = 3)
# inp = torch.rand(10,3,256,256)
# out = model(inp)
# print(out.size())

encoder = resnet18(in_channels = 3)
model = SupConProjection(encoder, 'mlp', 512, 200)
inp = torch.rand((10,3,32,32))
out = model(inp)
print(out.size())



# data_loader = data_loader(5) 
# print(len(data_loader.dataset))
# for idx, (images, labels) in enumerate(data_loader):
# 	labels_mask_similar_class = labels.unsqueeze(1).repeat(1, labels.shape[0]) == labels
# 	print(labels_mask_similar_class)