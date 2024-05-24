import torch
import torch.nn as nn
from utils import *
from gaitpt import GaitPT
from evaluators import *
from train import VanillaTrainer
import numpy as np
import random
from pytorch_metric_learning import losses

args = get_arguments()

# Set seed for reproducibility
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =================== Data ===================
train_transform, test_transform = get_transforms(args)
train_loader, test_loader = get_dataloaders(args, train_transform, test_transform)

# =================== Model ===================
model = GaitPT(args)
print("GaitPT - Number of Parameters: " + str(count_parameters(model)))

model= nn.DataParallel(model)
model.to(device)
if args.load_checkpoint_path:
	load_weights(model, args.load_checkpoint_path)

# =================== Helpers ===================
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr= args.lr / args.amplifier, max_lr=args.lr * args.amplifier, step_size_up=args.step_size, mode='exp_range', gamma=args.gamma, cycle_momentum=False)
criterion = losses.TripletMarginLoss(margin=args.margin)
trainer = VanillaTrainer(args, train_loader, device, optimizer, scheduler, criterion)

evaluator = get_evaluator(args, test_loader, device)

# =================== Train ===================
max_accuracy = 0
max_dataframe = None
for epoch in range(args.epochs):
	if args.run_type == 'train':
		print("Epoch: " + str(epoch))
		loss = trainer.train(model)

	if args.dataset == 'casia':
		accuracy_avg, scenario_accuracy, dataframe = evaluator.evaluate(model)
		print("Accuracy: " + str(accuracy_avg), end=" ")
		print("Max Accuracy: " + str(max_accuracy) + "\n")
		if max_accuracy < accuracy_avg:
			max_accuracy = accuracy_avg
			max_dataframe = dataframe
			save_weights(model, args.save_checkpoint_path)

	elif args.dataset in ['gait3d', 'grew']:
		r1, r5 = evaluator.evaluate(model)
		print("Max R1 Accuracy: " + str(max_accuracy) + "\n")
		if max_accuracy < r1:
			max_accuracy = r1
			max_dataframe = (r1, r5)
			save_weights(model, args.save_checkpoint_path)

	if args.run_type == 'test':
		break

# =================== Results ===================
print("Best Results")
if args.dataset == 'casia':
	print(max_dataframe.to_string())
else:
	print(max_dataframe)
print("Max Accuracy: " + str(max_accuracy))