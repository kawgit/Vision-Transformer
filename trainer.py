#    Copyright 2025 Kenneth Wilber (kawgit)

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import time
import torch
import wandb

from device import device
from settings import dataset_name

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
        self.after_batch_scheduler = Scheduler(after_batch, 10)
        self.after_epoch_scheduler = Scheduler(after_epoch, 30)

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

        wandb.init(
            project="transformer",
            config={
                "learning_rate": self.optimizer.param_groups[0]['lr'],
                "architecture": "transformer-{vocab_size}-{embedding_size}-{num_layers}x{layer_size}",
                "dataset": dataset_name,
                "epochs": num_epochs,
            }
        )

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

                loss = self.criterion(self.batch_outputs, self.batch_ys)
                
                loss.backward()
                self.optimizer.step()
                
                wandb.log({"loss": loss})
                
                self.batch_loss = loss.item()
                self.batch_losses.append(self.batch_loss)

                epoch_loss_total += self.batch_loss
                self.epoch_loss = epoch_loss_total / (self.batch_idx + 1)

                self.after_batch_scheduler(self)

            self.epoch_losses.append(self.epoch_loss)

            self.after_epoch_scheduler(self)

        print("Training completed")