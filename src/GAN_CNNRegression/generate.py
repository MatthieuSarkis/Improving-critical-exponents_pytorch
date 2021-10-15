# -*- coding: utf-8 -*-
#
# Written by Adel Sohbi, https://github.com/adelshb
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Generate images with the GAN model."""

from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # for ignoring the some of tf warnings

from dcgan import make_generator_model, make_discriminator_model

def main(args):

    generator = tf.keras.models.load_model(args.model_dir, compile=False)
    
    #generator = make_generator_model()
    #discriminator = make_discriminator_model()
    #generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    #discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    
    #checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
    #                                 discriminator_optimizer=discriminator_optimizer,
    #                                 generator=generator,
    #                                 discriminator=discriminator)
    
    #generator.load_weights('./data/models/gan-100k-2/training_checkpoints/ckpt-10')

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    # Generate and save images one by one
    for i in range(args.num):
        seed = tf.random.normal([1, args.noise_dim])
        gen = generator(seed, training=False)
        gen = tf.sign(gen)
        np.save('./data/generated/{}'.format(i), gen)

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--data_dir", type=str, default="./data/generated")
    parser.add_argument("--model_dir", type=str, default="./saved_models/gan_cnn")
    parser.add_argument("--noise_dim", type=int, default=100)

    args = parser.parse_args()
    main(args)
