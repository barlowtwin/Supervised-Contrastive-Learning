from resnet import resnet18
from SupCon import SupConProjection, Criterion, Classifier
from utils import custom_data_loader, train_SupCon_model, train_cross_entropy_model
from torchvision import datasets
import torch


batch_size = 256
in_channels = 3
projection_head_type = 'mlp'
encoder = resnet18(in_channels = in_channels) # outputs 512 features
in_dim = 512 # input to projection layer and output of encoder 
projection_dim = 128
temperature = 0.07
lr = 0.001
epochs = 100

if torch.cuda.is_available():
	device = torch.device("cuda")
	print("gpu detected for training")
else :
	device = torch.device("cpu")


model = SupConProjection(encoder, projection_head_type, in_dim, projection_dim)
model = model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)
criterion = Criterion(temperature = temperature, device = device)

data_loader = custom_data_loader(batch_size)
train(data_loader, model, criterion, optimizer,  device, epochs)

# # contrastive loss training is over

#########################################################################

# supervised training using cross entropy

model.encoder.freeze_layers()
classifier = Classifier(model, projection_dim, 10)
optimizer = torch.optim.SGD(model.parameters(), lr = lr)
classifier = classifier.to(device)
criterion = torch.nn.CrossEntropyLoss()
train_cross_entropy_model(data_loader, classifier, criterion, optimizer, device, epochs)






