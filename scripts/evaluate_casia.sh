#!/bin/bash

python main.py --train_data_path data/vitpose-train.csv --test_data_path data/vitpose-test.csv --epochs 200 --test_augmentations --test_aug_num_samples 2 --load_checkpoint_path checkpoints/casia-best.pth --run_type test