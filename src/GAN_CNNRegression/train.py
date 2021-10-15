# -*- coding: utf-8 -*-
#
# Written by Adel Sohbi and Matthieu Sarkis, https://github.com/adelshb, https://github.com/MatthieuSarkis
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Train the GAN model."""

from argparse import ArgumentParser
import os
from datetime import datetime

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # for ignoring the some of tf warnings

from dcgan import make_generator_model
from utils import *
from logger import Logger

def main(args):

    save_dir = os.path.join(args.save_dir, datetime.now().strftime("%Y.%m.%d.%H.%M.%S"))

    generator = make_generator_model(args.noise_dim)
    cnn = tf.keras.models.load_model(args.CNN_model_path, custom_objects={'tf': tf})
    
    loss_function = tf.keras.losses.MeanSquaredError()
    generator_optimizer = tf.keras.optimizers.Adam(1e-3)

    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, generator=generator)
    checkpoint_dir = os.path.join(save_dir, 'training_checkpoints')
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    logger = Logger(save_dir=save_dir)

    for epoch in range(args.epochs):

        logger.set_time_stamp(1)

        noise = tf.random.normal([args.batch_size, args.noise_dim], mean=0, stddev=1.0)

        gen_loss = train_step(generator= generator, 
                              cnn=cnn, 
                              generator_optimizer=generator_optimizer,  
                              loss_function=loss_function, 
                              noise= noise)

        if (epoch + 1) % args.ckpt_freq == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        logger.set_time_stamp(2)
        logger.logs['generator_loss'].append(gen_loss)
        logger.save_logs()
        logger.generate_plots(generator=generator,
                              cnn=cnn,
                              epoch=epoch,
                              labels="saved_models/CNN_L128_N10000/labels.json",
                              noise_dim=args.noise_dim,
                              bins_number=args.bins_number)
        logger.print_status(epoch=epoch)

    tf.keras.models.save_model(generator, save_dir)
    
    logger.save_metadata(vars(args))

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--bins_number", type=int, default=100)
    parser.add_argument("--noise_dim", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default="./saved_models/gan_cnn_regression")
    parser.add_argument("--ckpt_freq", type=int, default=10)
    parser.add_argument("--CNN_model_path", type=str, default="./saved_models/CNN_L128_N10000/saved-model.h5")

    args = parser.parse_args()
    main(args)
