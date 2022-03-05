from resnet import resnet18
from SupCon import SupConProjection, Criterion
from utils import custom_data_loader, train
import torch

batch_size = 256
in_channels = 3
projection_head_type = 'mlp'
encoder = resnet18(in_channels = in_channels) # outputs 512 features
in_dim = 512 # input to projection layer and output of encoder 
projection_dim = 128
temperature = 0.07
epochs = 100

if torch.cuda.is_available():
	device = torch.device("cuda")
else :
	device = torch.device("cpu")


model = SupConProjection(encoder, projection_head_type, in_dim, projection_dim)
model = model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)
criterion = Criterion(temperature = temperature, device = device)

data_loader = custom_data_loader(batch_size)
train(data_loader, model, criterion, optimizer,  device, epochs)