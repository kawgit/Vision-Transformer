import time
from device import device

class Scheduler:
    def __init__(self, action, delay):
        self.time_of_last_action = 0
        self.action = action
        self.delay = delay

    def __call__(self, trainer):
        firing = time.time() - self.time_of_last_action > self.delay
        
        if firing:
            self.time_of_last_action = time.time()
        
        if self.action:
            self.action(trainer, firing)

class Trainer:

    def __init__(self, model, dataloader, criterion, optimizer, after_batch, after_epoch):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.after_batch_scheduler = Scheduler(after_batch, .5)
        self.after_epoch_scheduler = Scheduler(after_epoch, 5)

        self.batch_xs = None
        self.batch_ys = None
        self.batch_outputs = None
        self.batch_index = None
        self.batch_loss = None
        self.epoch_index = None
        self.epoch_loss = None

        self.batch_losses = []
        self.epoch_losses = []

    def fit(self, num_epochs):

        self.model.train()
        self.epoch_losses = []

        for self.epoch_idx in range(num_epochs):

            epoch_loss_total = 0
            self.batch_losses = []

            for self.batch_idx, (self.batch_xs, self.batch_ys) in enumerate(self.dataloader):
                self.batch_xs = self.batch_xs.to(device)
                self.batch_ys = self.batch_ys.to(device)

                self.optimizer.zero_grad()

                self.batch_outputs = self.model(self.batch_xs)

                assert(self.batch_outputs.shape == self.batch_ys.shape)

                loss = self.criterion(self.batch_outputs, self.batch_ys)
                
                loss.backward()
                self.optimizer.step()
                
                self.batch_loss = loss.item()
                self.batch_losses.append(self.batch_loss)

                epoch_loss_total += self.batch_loss
                self.epoch_loss = epoch_loss_total / (self.batch_idx + 1)

                self.after_batch_scheduler(self)

            self.epoch_losses.append(self.epoch_loss)

            self.after_epoch_scheduler(self)

        print("Training completed")