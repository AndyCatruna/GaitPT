import torchvision.transforms as transforms
from datasets import *
from evaluators import *
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler
import numpy as np
import argparse
import yaml
import os

def get_evaluator(args, test_loader, device):
    if args.dataset == 'casia':
        return CasiaEvaluator(args, test_loader, device)
    elif args.dataset == 'gait3d':
        return Gait3DEvaluator(args, test_loader, device)
    elif args.dataset == 'grew':
        return GREWEvaluator(args, test_loader, device)

def get_transforms(args):
    transform_list = []
    if args.random_pace:
        transform_list.append(RandomPace())
    if args.random_pace_combination:
        transform_list.append(RandomPaceCombination())
    
    transform_list += [
            MirrorPoses(args.mirror_probability),
            FlipSequence(args.flip_probability),
            RandomSelectSequence(args.sequence_length),
            PointNoise(std=args.point_noise_std),
            JointNoise(std=args.joint_noise_std),
            CustomRepresentation(args),
    ]

    train_transform = transforms.Compose(transform_list)
    test_transform = transforms.Compose([
        SelectSequenceCenter(args.sequence_length),
        CustomRepresentation(args),
    ])

    if args.test_augmentations:
        test_transform = TestTimeAugmentation(transforms=test_transform, sequence_length=args.sequence_length, num_samples=args.test_aug_num_samples, use_flip=args.test_aug_flip, use_mirror=args.test_aug_mirror)

    return train_transform, test_transform

def get_dataloaders(args, train_transform, test_transform):
    if args.dataset == 'casia':
        trainset = CasiaDataset(args.train_data_path, sequence_length=args.sequence_length, transform=train_transform, train=True)
        testset = CasiaDataset(args.test_data_path, sequence_length=args.sequence_length, transform=test_transform)
        
    elif args.dataset == 'gait3d':
        trainset = Gait3DDataset(args.train_data_path, sequence_length=args.sequence_length, transform=train_transform, train=True)
        galleryset = Gait3DDataset('data/Gait3D-gallery.csv', sequence_length=args.sequence_length, transform=test_transform)
        probeset = Gait3DDataset('data/Gait3D-probe.csv', sequence_length=args.sequence_length, transform=test_transform)

    elif args.dataset == 'grew':
        trainset = GREWDataset(args.train_data_path, sequence_length=args.sequence_length, transform=train_transform, train=True)
        galleryset = GREWDataset('data/GREW-gallery-val.csv', sequence_length=args.sequence_length, transform=test_transform)
        probeset = GREWDataset('data/GREW-probe-val.csv', sequence_length=args.sequence_length, transform=test_transform)

    num_identities = 74 if args.dataset == 'casia' else args.num_identities
    full_labels_list = torch.tensor([sample['label'][0] for sample in trainset])
    balanced_sampler = BalancedBatchSampler(labels=full_labels_list, n_classes=num_identities, n_samples=args.samples_per_batch)

    train_loader = DataLoader(
        trainset,
        num_workers=args.num_workers,
        pin_memory=True,
        batch_sampler=balanced_sampler,
    )

    if args.dataset == 'casia':
        test_loader = DataLoader(
            testset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
    elif args.dataset in ['gait3d', 'grew']:
        gallery_loader = DataLoader(
            galleryset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

        probe_loader = DataLoader(
            probeset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

        test_loader = (gallery_loader, probe_loader)

    return train_loader, test_loader

class BalancedBatchSampler(BatchSampler):
    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_weights(model, checkpoint_path='checkpoints/checkpoint.pth'):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    print('Saving weights to checkpoint...\n')
    torch.save(model.state_dict(), checkpoint_path)

def load_weights(model, checkpoint_path='checkpoints/checkpoint.pth'):
    if os.path.exists(checkpoint_path):
        print('Loading weights from checkpoint...\n')
        model.load_state_dict(torch.load(checkpoint_path))

def get_arguments():

    parser = argparse.ArgumentParser()
    
    # Learning Hyperparameters
    parser.add_argument('--epochs', type=int, default=300)    
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--scheduler', type=str, choices = ['step', 'onecycle', 'cosine', 'cyclic', 'none'], default='cyclic')
    parser.add_argument('--step_size', type=int, default=15)
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--amplifier', type=float, default=10)
    parser.add_argument('--criterion', type=str, choices = ['triplet', 'supcon', 'arcface', 'circle'], default='triplet')
    parser.add_argument('--margin', type=float, default=0.01)

    # Augmentations
    parser.add_argument("--point_noise_std", type=float, default=0.05)
    parser.add_argument("--joint_noise_std", type=float, default=0)
    parser.add_argument("--flip_probability", type=float, default=0)
    parser.add_argument("--mirror_probability", type=float, default=0)
    parser.add_argument("--random_pace", action='store_true')
    parser.add_argument("--random_pace_combination", action='store_true')
    parser.add_argument("--angle_mix", action='store_true')
    parser.add_argument("--scenario_mix", action='store_true')
    parser.add_argument("--rm_conf", type=bool, default=True)
    parser.add_argument("--test_augmentations", action='store_true')
    parser.add_argument("--test_aug_num_samples", type=int, default=2)
    parser.add_argument("--test_aug_flip", action='store_true')
    parser.add_argument("--test_aug_mirror", action='store_true')

    # Dataset
    parser.add_argument("--dataset", type=str, choices = ['casia', 'grew', 'gait3d'], default='casia')
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--sequence_length", type=int, default=60)
    parser.add_argument("--samples_per_batch", type=int, default=6)
    parser.add_argument("--num_identities", type=int, default=200)

    # Others
    parser.add_argument('--config_file', type=str, default='configs/gaitpt_config.yaml')
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--seed', type=int, default=20)
    parser.add_argument('--save_checkpoint_path', type=str, default='checkpoints/checkpoint.pth')
    parser.add_argument('--load_checkpoint_path', type=str, default='')
    parser.add_argument('--run_type', type=str, choices = ['train', 'test'], default='train')

    args = parser.parse_args()

    # Load config file
    with open(args.config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Update args with config file
    args.__dict__.update(config)

    return args
