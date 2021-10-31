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

from torch.serialization import save
#from torch.utils.data import DataLoader, random_split
#from torchsummary import summary

from src.data import generate_data_torch
from src.cnn.network import CNN

def main(args):

    save_dir = os.path.join(args.save_dir, datetime.now().strftime("%Y.%m.%d.%H.%M.%S"))

    X_train, y_train, X_test, y_test = generate_data_torch(dataset_size=args.dataset_size,
                                                           lattice_size=args.lattice_size,
                                                           p_list=None,
                                                           split=True,
                                                           save_dir=None)
    
    model = CNN(lattice_size=args.lattice_size,
                n_conv_layers=4,
                n_dense_layers=3,
                n_neurons=512,
                dropout_rate=0.0,
                learning_rate=10e-4,
                device=args.device,
                save_dir=save_dir)
    
    model.train(epochs=args.epochs,
                 X_train=X_train,
                 y_train=y_train,
                 X_test=X_test,
                 y_test=y_test,
                 batch_size=args.batch_size,
                 save_checkpoints=args.save_checkpoints)
    
    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
        
    with open(os.path.join(save_dir, 'loss.json'), 'w') as f:
        json.dump(model.loss_history, f, indent=4)

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
    
    parser.add_argument('--save_checkpoints', dest='save_checkpoints', action='store_true')
    parser.add_argument('--no-save_checkpoints', dest='save_checkpoints', action='store_false')
    parser.set_defaults(save_checkpoints=True)
    
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
