import numpy as np
import pandas as pd
import torch
import sys
from sklearn.neighbors import KNeighborsClassifier

class CasiaEvaluator():
	def __init__(self, args, data_loader, device):
		self.args = args
		self.device = device
		self.data_loader = data_loader

	def get_embeddings(self, model, data_loader):
		model.eval()

		with torch.no_grad():
			all_info = []
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

				ids, scenarios, vid_nums, angles = data['label']
				info = list(zip(ids, scenarios, vid_nums, angles))

				all_info += info
				output = model(poses)
				if self.args.test_augmentations:
					output = torch.mean(output.view(num_samples, bsz, -1), dim=0)
				all_embeddings.extend(output.cpu().numpy())

		return all_embeddings, all_info
	
	def compute_results(self, embeddings, info):
		identities, scenarios, scenario_indices, angles = zip(*info)
		identities = np.array(identities)
		scenarios = np.array(scenarios)
		scenario_indices = np.array(scenario_indices)
		
		angles = np.array(angles)
		embeddings = np.array(embeddings)

		all_angles = [*range(0, 181, 18)]
		accuracies = np.zeros((3, 11, 11))

		galleries = dict()
		probes = dict()
		for scenario in range(3):
			for probe_angle in all_angles:
				for gallery_angle in all_angles:
					if probe_angle == gallery_angle:
						continue

					gallery_indices = (angles == gallery_angle) & (scenarios == 0) & (scenario_indices <= 4)
					probe_indices = (angles == probe_angle) & (scenarios == scenario)
					if scenario == 0:
						probe_indices = probe_indices & (scenario_indices > 4)
					
					current_info = (scenario, probe_angle, gallery_angle)
					galleries[current_info] = gallery_indices.copy()
					probes[current_info] = probe_indices.copy()

		for keys in galleries.keys():
			gallery_indices = galleries[keys]
			gallery_embeddings = embeddings[gallery_indices]
			gallery_identities = identities[gallery_indices]

			metric = 'minkowski'
			if self.args.criterion != 'triplet':
				metric = 'cosine'
			
			classifier = KNeighborsClassifier(n_neighbors=1, metric=metric)
			classifier.fit(gallery_embeddings, gallery_identities)

			probe_indices = probes[keys]
			probe_embeddings = embeddings[probe_indices]    
			probe_identities = identities[probe_indices]

			predictions = classifier.predict(probe_embeddings)
			
			current_acc = np.sum(predictions == probe_identities) / len(probe_identities)
			
			accuracies[keys[0], keys[1] // 18, keys[2] // 18] = current_acc
		
		per_probe_accuracies = np.sum(accuracies, axis=2) / 10

		per_scenario_accuracies = np.mean(per_probe_accuracies, axis=1)
		accuracy_avg = np.mean(per_scenario_accuracies)

		print_accuracies = np.concatenate((per_probe_accuracies, np.expand_dims(per_scenario_accuracies, axis=1)), axis=1)
		scenario_index = ['NM', 'BG', 'CL']
		dataframe = pd.DataFrame(print_accuracies)
		dataframe.set_index([scenario_index], inplace=True)
		angles = [str(angle) for angle in all_angles]
		angles += ['Mean']
		dataframe.columns = angles

		print("=" * 150)
		print(dataframe.to_string())
		print("=" * 150)

		return accuracy_avg, per_scenario_accuracies, dataframe

	def evaluate(self, model):
		embeddings, info = self.get_embeddings(model, self.data_loader)

		accuracy_avg, per_scenario_accuracies, dataframe = self.compute_results(embeddings, info)
		
		return accuracy_avg, per_scenario_accuracies, dataframe