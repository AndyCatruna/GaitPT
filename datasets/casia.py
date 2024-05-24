import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from .augmentation import mix_sequences, ToTensor
scenario_to_label = {'nm': 0, 'bg': 1, 'cl': 2}

class CasiaDataset(Dataset):
    def __init__(self, data_path, sequence_length=60, transform=None, train=False, angle_mix=0., scenario_mix=0.):
        self.train = train
        self.poses, self.info = self.initialize_data(data_path, sequence_length)
        self.search_info = np.array(self.info)
        self.transform = transform
        self.angle_mix = angle_mix
        self.scenario_mix = scenario_mix
        self.to_tensor = ToTensor()
        print("Number of samples: ", len(self.poses))

    def get_info(self, image_name):
        current_info = image_name.split('/')[1]
        current_info = current_info.split('-')
        identity = int(current_info[0])
        scenario = scenario_to_label[current_info[1]]
        video_num = int(current_info[2])
        angle = int(current_info[3])

        return (identity, scenario, video_num, angle)

    def get_frame_num(self, image_name):
        return int(image_name.split('/')[-1][:-4])

    def initialize_data(self, data_path, sequence_length=60):
        data = pd.read_csv(data_path)
        
        all_poses = []
        all_info = []

        image_names = data['image_name']
        info = [self.get_info(image_name) for image_name in image_names]
        frame_nums = [self.get_frame_num(image_name) for image_name in image_names]

        data['info'] = info
        data['frame_num'] = frame_nums
        
        all_data = data.groupby('info', sort=False)
        for _, group in all_data:
            current_sequence = group.iloc[:, 1:-2].values.reshape(-1, 17, 3)
            if len(current_sequence) <= sequence_length:
                continue

            all_poses.append(current_sequence)
            all_info.append(group.iloc[0]['info'])

        # Oversample from CL and BG scenarios
        if self.train:
            for i in range(len(all_poses)):
                current_info = all_info[i]
                if current_info[1] != 0:
                    all_poses.append(all_poses[i])
                    all_info.append(current_info)

                    all_poses.append(all_poses[i])
                    all_info.append(current_info)

        return all_poses, all_info

    def __len__(self):
        return len(self.poses)

    def get_similar_direction(self, info):
        angle = info[3]
        angle_list = [angle]
        if angle == 0:
            angle_list.append(18)
        elif angle == 180:
            angle_list.append(162)
        else:
            angle_list.append(angle + 18)
            angle_list.append(angle - 18)

        mask = np.isin(self.search_info[:, 3], angle_list) & (self.search_info[:, 0] == info[0]) & (self.search_info[:, 1] == info[1]) & (self.search_info[:, 2] != info[2])
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return None

        random_index = np.random.choice(indices)
        new_sequence = self.poses[random_index]

        return new_sequence

    def get_different_scenario(self, info):
        scenario = info[1]
        scenario_list = [0, 1, 2]
        scenario_list.remove(scenario)

        mask = np.isin(self.search_info[:, 1], scenario_list) & (self.search_info[:, 0] == info[0]) & (self.search_info[:, 3] == info[3])
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return None

        random_index = np.random.choice(indices)
        new_sequence = self.poses[random_index]

        return new_sequence

    def scenario_mix_sequences(self, sequence, info):
        new_sequence = self.get_similar_direction(info)
        if new_sequence is not None:
            sequence = mix_sequences(sequence, new_sequence)
        
        return sequence

    def angle_mix_sequences(self, sequence, info):
        new_sequence = self.get_similar_direction(info)
        if new_sequence is not None:
            sequence = mix_sequences(sequence, new_sequence)
        
        return sequence

    def normalize_width(self, poses):
        poses /= 320

        return poses
    
    def process_poses(self, poses):
        poses = poses[:, :, :2]
        poses = self.normalize_width(poses)
        poses = self.to_tensor(poses)

        return poses
    
    def __getitem__(self, idx):
        poses = self.poses[idx]
        info = self.info[idx]

        if self.train and np.random.rand() < self.angle_mix:
            poses = self.angle_mix_sequences(poses, info)
        if self.train and np.random.rand() < self.scenario_mix:
            poses = self.scenario_mix_sequences(poses, info)
        
        poses = np.float32(poses)
        if self.transform:
            poses = self.transform(poses)

        if self.train:
            poses = self.process_poses(poses)
        else:
            poses = [self.process_poses(sample_poses) for sample_poses in poses]

        output = {'pose': poses, 'label': info}

        return output
