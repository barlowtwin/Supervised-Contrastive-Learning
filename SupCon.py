import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConProjection(nn.Module):

	# encoder : resnet18, resnet34
	# head_type : 'linear' or 'mlp'
	# proj_dim : dim of Projection

	def __init__(self, encoder, head_type, in_dim, proj_dim):
		super(SupConProjection, self).__init__()

		self.encoder = encoder

		if head_type == 'linear' :
			self.head = nn.Linear(in_dim, proj_dim)
		elif head_type == 'mlp':
			self.head = nn.Sequential(
				nn.Linear(in_dim, in_dim),
				nn.ReLU(inplace = True),
				nn.Linear(in_dim, proj_dim))
		else :
			raise NotImplementedError('head type not supported : {}'.format(head))

	def forward(self, x):
		out = self.encoder(x)
		return F.normalize(self.head(out), dim = 1)



class Classifier(nn.Module):

	def __init__(self, model, proj_dim,  num_classes):
		super(Classifier, self).__init__()

		self.model = model # batch_size x 200
		self.linear = nn.Linear(proj_dim, num_classes) # 200 x num_classes

	def forward(self, input): # input :  batch of images
		features = self.model(input) 
		return self.linear(features) # batch_size x num_classes



class Criterion(nn.Module):

	def __init__(self, temperature = 0.07, device = None):
		super(Criterion, self).__init__()
		self.temperature = temperature
		self.device = device

	def forward(self, projections, labels): 

		# 128 is number of features
		# projections : 2*batch_size x 128 ; 2 stands for features from 2 images (real and augmented)
		# labels : 2*batch_size

		dot_product = torch.mm(projections, projections.T) / self.temperature # batch_size x batch_size
		exp_dot_product = torch.exp(dot_product - torch.max(dot_product, dim = 1)[0]) + 1e-5 # 1e-5 added to avoid log0 # 20 x 20

		mask_similar_classes = (labels.unsqueeze(1).repeat(1, labels.shape[0]) == labels).to(self.device) # batch_size x batch_size boolean tensor
		# classes similar to the considered class are True and rest are False

		mask_anchor = 1 - torch.eye(labels.shape[0]).to(self.device) # 20 x 20
		# all except diagonal elements are 0

		mask_combined = mask_similar_classes * mask_anchor # 20 x 20
		# elements belonging to same class are 0 

		cardinality_per_samples = torch.sum(mask_combined, dim = 1)

		log_prob = - torch.log(exp_dot_product) / (torch.sum(exp_dot_product * mask_anchor, dim = 1, keepdim = True))
		loss_per_sample = torch.sum(log_prob * mask_combined, dim = 1) / cardinality_per_samples
		loss_mean = torch.mean(loss_per_sample)

		return loss_mean











		













