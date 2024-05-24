import numpy as np
import torch
import random
import math

class ToTensor(object):
	def __call__(self, data):
		return torch.tensor(data, dtype=torch.float)

class FlipSequence(object):
	def __init__(self, probability=0.5):
		self.probability = probability

	def __call__(self, data):
		if np.random.random() <= self.probability:
			return np.flip(data, axis=0).copy()
		return data

class MirrorPoses(object):
	def __init__(self, probability=0.5):
		self.probability = probability

	def __call__(self, data):
		if np.random.random() <= self.probability:
			center = np.mean(data[:, :, 0], axis=1, keepdims=True)
			data[:, :, 0] = center - data[:, :, 0] + center

		return data

class RandomSelectSequence(object):
	def __init__(self, sequence_length=10):
		self.sequence_length = sequence_length

	def __call__(self, data):
		start = np.random.randint(0, data.shape[0] - self.sequence_length)
		end = start + self.sequence_length
		return data[start:end]

class SelectSequenceCenter(object):
	def __init__(self, sequence_length=10):
		self.sequence_length = sequence_length

	def __call__(self, data):
		start = int((data.shape[0]/2) - (self.sequence_length / 2))
		end = start + self.sequence_length
		return data[start:end]

class PointNoise(object):
	"""
	Add Gaussian noise to pose points
	std: standard deviation
	"""

	def __init__(self, std=0.15):
		self.std = std

	def __call__(self, data):
		noise = np.random.normal(0, self.std, data.shape).astype(np.float32)
		return data + noise

class JointNoise(object):
	"""
	Add Gaussian noise to joint
	std: standard deviation
	"""

	def __init__(self, std=0.5):
		self.std = std

	def __call__(self, data):
		# T, V, C
		noise = np.hstack((
			np.random.normal(0, 0.25, (data.shape[1], 2)),
			np.zeros((data.shape[1], 1))
		)).astype(np.float32)

		return data + np.repeat(noise[np.newaxis, ...], data.shape[0], axis=0)

class CustomRepresentation:
	def __init__(self, args):
		if args.stage3_partition in ['head-upper-lower', 'all']:
			self.indices = [0, 1, 3, 17, 2, 4, 5, 7, 9, 6, 8, 10, 11, 13, 15, 12, 14, 16]
		elif args.stage3_partition == 'head-left-right':
			self.indices = [0, 1, 3, 17, 2, 4, 5, 7, 9, 11, 13, 15, 6, 8, 10, 12, 14, 16]
		elif args.stage3_partition == 'head-opposite':
			self.indices = [0, 1, 3, 17, 2, 4, 5, 7, 9, 12, 14, 16, 6, 8, 10, 11, 13, 15]

	def __call__(self, data):
		heads = data[:,0,:]
		heads = np.expand_dims(heads, 1)
		data = np.append(data, heads, axis=1)
		data = data[:,self.indices,:]

		return data

class TestTimeAugmentation:
	def __init__(self, transforms, sequence_length=60, num_samples=1, use_flip=False, use_mirror=False):
		self.sequence_length = sequence_length
		self.num_samples = num_samples
		
		self.flip = None
		if use_flip:
			self.flip = FlipSequence(probability=1)
		
		self.mirror = None
		if use_mirror:
			self.mirror = MirrorPoses(probability=1)
		
		self.transforms = transforms

	def __call__(self, data):
		if self.num_samples == 1:
			all_data = [data]
			if self.flip is not None:
				flipped_data = self.flip(data)
				all_data += [flipped_data]
			
			if self.mirror is not None:
				mirrored_data = self.mirror(data)
				all_data += [mirrored_data]

		if self.num_samples == 2:
			input_length = data.shape[0]
			sequence_center = input_length // 2
			start1 = 0
			end1 = max(sequence_center, start1 + self.sequence_length)

			end2 = input_length - 1
			start2 = min(sequence_center, end2 - self.sequence_length)

			data1 = data[start1:end1]
			data2 = data[start2:end2]

			all_data = [data1, data2]
			if self.flip is not None:
				flipped_data1 = self.flip(data1)
				flipped_data2 = self.flip(data2)
				all_data += [flipped_data1, flipped_data2]

			if self.mirror is not None:
				mirrored_data1 = self.mirror(data1)
				mirrored_data2 = self.mirror(data2)
				all_data += [mirrored_data1, mirrored_data2]
		
		data = [self.transforms(d) for d in all_data]

		return data

def interpolate(sequence, repeat=False):
	if repeat:
		return np.repeat(sequence, 2, axis=0)
	
	h, w = sequence.shape[:2]
	new_sequence = []
	
	for i in range(h - 1):
		j = i + 1
		seq1 = sequence[i]
		seq2 = sequence[j]

		inter_seq = (seq1 + seq2) / 2

		new_sequence += [seq1, inter_seq]

	return np.array(new_sequence)

def transform_to_pace(sequence, pace):
	if pace == 1:
		return sequence
	elif pace < 1:
		for _ in range(int(0.5 // pace)):
			sequence = interpolate(sequence)
		
		return sequence
	else:
		return sequence[::pace]

class RandomPace(object):
	def __init__(self, paces = [0.5, 1, 1, 1, 2], minimum_length=60):
		self.paces = paces
		self.minimum_length = minimum_length

	def __call__(self, sequence):
		h, w = sequence.shape[:2]
		
		pace = random.choice(self.paces)
		new_sequence = transform_to_pace(sequence, pace)
		
		while new_sequence.shape[0] < self.minimum_length:
			new_sequence = interpolate(new_sequence)

		return new_sequence

class RandomPaceCombination(object):
	def __init__(self, paces = [0.5, 0.5, 1, 1, 1, 1, 1, 2, 2], minimum_length=60):
		self.paces = paces
		self.minimum_length = minimum_length

	def __call__(self, sequence):
		h, w = sequence.shape[:2]

		number_of_paces = random.randint(1, 3)
		
		new_sequence = None
		slice_size = h // number_of_paces
		
		for count in range(number_of_paces):
			pace = random.choice(self.paces)
			slice = sequence[count * slice_size: (count + 1) * slice_size]

			new_slice = transform_to_pace(slice, pace)
			if new_sequence is None:
				new_sequence = new_slice
			else:
				new_sequence = np.concatenate((new_sequence, new_slice), axis=0)

		while new_sequence.shape[0] < self.minimum_length:
			new_sequence = interpolate(new_sequence)

		return new_sequence

def mix_sequences(sequence1, sequence2):
	alpha = random.random()
	mixed_length = min(sequence1.shape[0], sequence2.shape[0])
	sequence1 = sequence1[:mixed_length]
	sequence2 = sequence2[:mixed_length]

	return alpha * sequence1 + (1 - alpha) * sequence2