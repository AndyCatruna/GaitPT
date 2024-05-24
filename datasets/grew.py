import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from .augmentation import ToTensor
import torchvision.transforms as transforms

class GREWDataset(Dataset):
    def __init__(self, data_path, sequence_length=30, transform=None, train=False):
        self.train = train
        self.poses, self.info = self.initialize_data(data_path, sequence_length)
        self.search_info = np.array(self.info)
        self.transform = transform
        self.to_tensor = ToTensor()
        print("Number of sequences: ", len(self.poses))

    def get_info(self, image_name):
        current_info = image_name.split('/')[:-1]
        current_info = "-".join(current_info)
        
        return current_info

    def get_frame_num(self, image_name):
        frame_num = int(image_name.split('/')[-1][:-4].split('_')[-3]) - 1
        return frame_num

    def initialize_data(self, data_path, sequence_length=30):
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
            current_sequence = group.iloc[:, 4:-2].values.reshape(-1, 17, 3)
            if len(current_sequence) <= sequence_length:
                continue

            all_poses.append(current_sequence)
            all_info.append((group.iloc[0]['identity'], group.iloc[0]['image_width'], group.iloc[0]['image_height'], group.iloc[0]['info']))

        return all_poses, all_info

    def normalize_height(self, poses, info):
        poses[:, :, :2] /= info[2]

        return poses
    
    def process_poses(self, poses, info):
        poses = poses[:, :, :2]
        poses = self.normalize_height(poses, info)
        poses = self.to_tensor(poses)

        return poses
    
    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        poses = self.poses[idx]
        info = self.info[idx]

        poses = np.float32(poses)
        if self.transform:
            poses = self.transform(poses)

        if self.train:
            poses = self.process_poses(poses, info)
        else:
            poses = [self.process_poses(sample_poses, info) for sample_poses in poses]

        output = {'pose': poses, 'label': info}

        return output
