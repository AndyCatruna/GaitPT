import numpy as np

class VanillaTrainer():
    def __init__(self, args, train_loader, device, optimizer, scheduler, criterion):
        self.args = args
        self.train_loader = train_loader
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

    def train(self, model):
        model.train()
        total_train_loss = 0

        for data in self.train_loader:
            poses = data['pose'].to(self.device)
            labels = data['label'][0].to(self.device)

            self.optimizer.zero_grad()
            features = model(poses)
            loss = self.criterion(features, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip)

            self.optimizer.step()

            total_train_loss += loss.item()

        self.scheduler.step()

        train_loss = np.round(total_train_loss, 4)
        
        print("TRAIN LOSS: " + str(train_loss))
        
        return train_loss
