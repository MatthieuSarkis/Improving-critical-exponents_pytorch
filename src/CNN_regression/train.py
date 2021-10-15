# -*- coding: utf-8 -*-
#
# Written by Matthieu Sarkis, https://github.com/MatthieuSarkis
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from argparse import ArgumentParser
from datetime import datetime
import json
import os
import time
import torch
from torch.utils.data import DataLoader, random_split
from torch import optim
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummary import summary
from typing import Dict, Tuple

from src.CNN_regression.data import generate_data_torch
from src.CNN_regression.network import cnn


def main(args):

    # Create the directory tree
    save_dir = os.path.join(args.save_dir, datetime.now().strftime("%Y.%m.%d.%H.%M.%S"))
    save_dir_model = os.path.join(save_dir, 'model')
    os.makedirs(save_dir_model, exist_ok=True)
    
    # Create the data
    X_train, y_train, X_test, y_test = generate_data_torch(args.dataset_size)
    
    # Create the model, optimizer, loss function and callbacks 
    model = cnn(lattice_size=args.lattice_size,
                n_conv_layers=4,
                n_dense_layers=3,
                n_neurons=512,
                dropout_rate=0.0,
                device=args.device)
    
    optimizer = optim.Adam(params=model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    if args.set_lr_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=False)
    
    # Training loop
    model, loss_history = train(epochs=args.epochs,
                                X_train=X_train,
                                y_train=y_train,
                                X_test=X_test,
                                y_test=y_test,
                                batch_size=args.batch_size,
                                model=model,
                                optimizer=optimizer,
                                criterion=criterion,
                                scheduler=scheduler,
                                device=args.device,
                                save_dir_model=save_dir_model)
    
    if args.save_model:
        torch.save(model.state_dict(), os.path.join(save_dir_model, 'final_model.pt'))

    # Save a few logs
    if args.set_save_args:
        with open(os.path.join(save_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)
        
    if args.dump_loss:
        with open(os.path.join(save_dir, 'loss.json'), 'w') as f:
            json.dump(loss_history, f, indent=4)


def train(epochs: int,
          X_train: torch.tensor,
          y_train: torch.tensor,
          X_test: torch.tensor,
          y_test: torch.tensor,
          batch_size: int,
          model: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          criterion: torch.nn.modules.loss._Loss,
          scheduler: torch.optim.lr_scheduler._LRScheduler,
          device: str,
          save_dir_model: str,
          ) -> Tuple[cnn, Dict[list, list]]:
    
    loss_history = {'train': [], 'test': []}
    
    for epoch in range(epochs):
        
        initial_time = time.time()
        
        permutation = torch.randperm(X_train.shape[0])
        train_loss = 0.0
        model.train()
        for i in range(0, X_train.shape[0], batch_size):

            indices = permutation[i:i+batch_size]

            inputs = X_train[indices].to(device)
            labels = y_train[indices].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            optimizer.step()
            train_loss += loss.item()

        permutation = torch.randperm(X_test.shape[0])
        test_loss = 0.0
        model.eval()
        for i in range(0, X_test.shape[0], batch_size):

            indices = permutation[i:i+batch_size]

            inputs = X_test[indices].to(device)
            labels = y_test[indices].to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
    
        train_loss /= (X_train.shape[0]//batch_size)
        test_loss /= (X_test.shape[0]//batch_size)
        
        loss_history['train'].append(train_loss)
        loss_history['test'].append(test_loss)
        
        if args.set_lr_scheduler:
            scheduler.step(test_loss)
        
        print("Epoch: {}/{}, Train Loss: {:.4f}, Test Loss: {:.4f}, Time: {:.2f}s".format(epoch+1, epochs, train_loss, test_loss, time.time()-initial_time))

        os.makedirs(save_dir_model, exist_ok=True)

        if args.save_checkpoints:
            checkpoint_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
                }
            torch.save(checkpoint_dict, os.path.join(save_dir_model, 'ckpt_{}.pt'.format(epoch)))

    return model, loss_history


if __name__ == '__main__':

    parser = ArgumentParser()
    
    parser.add_argument("--save_dir", type=str, default='./saved_models/cnn_regression')
    parser.add_argument("--lattice_size", type=int, default=128)
    parser.add_argument("--dataset_size", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--learning_rate", type=float, default=10e-4)
    parser.add_argument("--device", type=str, default='cpu')
    
    parser.add_argument('--set_save_args', dest='set_save_args', action='store_true')
    parser.add_argument('--no-set_save_args', dest='set_save_args', action='store_false')
    parser.set_defaults(set_save_args=True)
    
    parser.add_argument('--set_lr_scheduler', dest='set_lr_scheduler', action='store_true')
    parser.add_argument('--no-set_lr_scheduler', dest='set_lr_scheduler', action='store_false')
    parser.set_defaults(set_lr_scheduler=True)

    parser.add_argument('--save_checkpoints', dest='save_checkpoints', action='store_true')
    parser.add_argument('--no-save_checkpoints', dest='save_checkpoints', action='store_false')
    parser.set_defaults(save_checkpoints=True)

    parser.add_argument('--save_model', dest='save_model', action='store_true')
    parser.add_argument('--no-save_model', dest='save_model', action='store_false')
    parser.set_defaults(save_model=True)

    parser.add_argument('--dump_loss', dest='dump_loss', action='store_true')
    parser.add_argument('--no-dump_loss', dest='dump_loss', action='store_false')
    parser.set_defaults(dump_loss=True)

    args = parser.parse_args()
    main(args)




### If one wants to use custom datasets

    # Generate the train and test datasets
    #train_data = LatticeConfigurations(dataset_size=args.dataset_size,
    #                                   lattice_size=args.lattice_size)
#
    #train_set, test_set = random_split(train_data, [args.dataset_size-args.dataset_size//4, args.dataset_size//4])
#
    #train_loader = DataLoader(dataset=train_set,
    #                          batch_size=BATCH_SIZE,
    #                          shuffle=True)
#
    #test_loader = DataLoader(dataset=test_set,
    #                         batch_size=BATCH_SIZE,
    #                         shuffle=True)


    #print('\n', summary(model, (1, 128, 128)), '\n')
    
    # Training loop
    #loss_history = {'train': [], 'Test': []}
    #for epoch in range(EPOCHS):
    #    
    #    initial_time = time.time()
    #    
    #    train_loss = 0.0
    #    model.train()
    #    for sample in train_loader:
#
    #        inputs = sample['images'].to(device)
    #        labels = sample['labels'].to(device)
    #        
    #        optimizer.zero_grad()
    #        outputs = model(inputs)
    #        loss = criterion(outputs, labels)
    #        loss.backward()
    #        
    #        optimizer.step()
    #        train_loss += loss.item()
#
    #    test_loss = 0.0
    #    model.eval()
    #    for sample in test_loader:
#
    #        inputs = sample['images'].to(device)
    #        labels = sample['labels'].to(device)
    #        
    #        outputs = model(inputs)
    #        loss = criterion(outputs, labels)
    #        test_loss += loss.item()
    #
    #    train_loss /= len(train_loader)
    #    test_loss /= len(test_loader)
    #    
    #    loss_history['train'].append(train_loss)
    #    loss_history['Test'].append(test_loss)
    #    
    #    if args.set_lr_scheduler:
    #        scheduler.step(test_loss)
    #    
    #    print("Epoch: {}/{}, Train Loss: {:.4f}, Test Loss: {:.4f}, Time: {:.2f}s".format(epoch+1, EPOCHS, train_loss, test_loss, time.time()-initial_time))
#
    #    if args.save_checkpoints:
    #        checkpoint_dict = {
    #            'epoch': epoch,
    #            'model_state_dict': model.state_dict(),
    #            'optimizer_state_dict': optimizer.state_dict(),
    #            'train_loss': train_loss,
    #            'test_loss': test_loss,
    #            }
    #        torch.save(checkpoint_dict, os.path.join(save_dir_model, 'ckpt_{}.pt'.format(epoch)))
