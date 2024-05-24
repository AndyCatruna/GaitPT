#!/bin/bash

python main.py --train_data_path data/GREW-train.csv --test_data_path data/GREW-gallery.csv --dataset grew --sequence_length 30 --test_augmentations --test_aug_num_samples 2 --samples_per_batch 4 --num_identities 200 --load_checkpoint_path checkpoints/grew-checkpoint.pth --run_type test