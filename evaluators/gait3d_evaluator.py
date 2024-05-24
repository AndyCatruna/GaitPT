import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import KNeighborsClassifier

class Gait3DEvaluator():
	def __init__(self, args, data_loader, device):
		self.args = args
		self.device = device
		self.gallery_loader, self.probe_loader = data_loader

	def get_embeddings(self, model, data_loader):
		model.eval()

		with torch.no_grad():
			all_identities = []
			all_embeddings = []
			for data in data_loader:
				if self.args.test_augmentations:
					num_samples = len(data['pose'])
					poses = data['pose'][0].to(self.device)
					bsz = poses.shape[0]
					for i in range(1, num_samples):
						poses = torch.cat((poses, data['pose'][i].to(self.device)), dim=0)
				else:
					poses = data['pose'].to(self.device)

				ids, _, _, _ = data['label']

				all_identities += ids
				output = model(poses)
				if self.args.test_augmentations:
					output = torch.mean(output.view(num_samples, bsz, -1), dim=0)
				all_embeddings.extend(output.cpu().numpy())

		return all_embeddings, all_identities
	
	def compute_results(self, gallery_embeddings, gallery_identities, probe_embeddings, probe_identities):
		gallery_identities = np.array(gallery_identities)
		probe_identities = np.array(probe_identities)
	
		gallery_embeddings = np.array(gallery_embeddings)
		probe_embeddings = np.array(probe_embeddings)


		metric = 'minkowski'
		if self.args.criterion != 'triplet':
			metric = 'cosine'
		
		# Rank 1 Accuracy
		classifier = KNeighborsClassifier(n_neighbors=1, metric=metric)
		classifier.fit(gallery_embeddings, gallery_identities)

		predictions = classifier.predict(probe_embeddings)
		
		r1 = np.sum(predictions == probe_identities) / len(probe_identities)
		
		# Rank 5 Accuracy
		classifier = KNeighborsClassifier(n_neighbors=5, metric=metric)
		classifier.fit(gallery_embeddings, gallery_identities)

		top5_predictions = classifier.kneighbors(probe_embeddings, return_distance=False)
		top5_predictions = np.array([gallery_identities[pred] for pred in top5_predictions])
		
		r5 = ((top5_predictions == probe_identities.reshape(-1, 1)).sum(axis=1) > 0).sum() / len(probe_identities)

		print('R-1 Accuracy: ' + str(r1))
		print('R-5 Accuracy: ' + str(r5))

		return r1, r5

	def evaluate(self, model):
		gallery_embeddings, gallery_info = self.get_embeddings(model, self.gallery_loader)
		probe_embeddings, probe_info = self.get_embeddings(model, self.probe_loader)

		r1, r5 = self.compute_results(gallery_embeddings, gallery_info, probe_embeddings, probe_info)
		
		return r1, r5