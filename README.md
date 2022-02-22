# Improving critical exponent estimations with Generative Networks

## Requirements

* c++17
* Python 3.8+
* numpy
* torch

```shell
pip install -e .
```

### Generating Ising configurations

Modify src/data_factory/ising/params.h as needed, then compile and run as follows:

```shell
g++ src/data_factory/ising/main.cpp -std=c++17 -O3 -o src/data_factory/ising/ising
./src/data_factory/ising/ising
```

## Control parameter estimation with Convolutional Neural Networks

```shell
python src/cnn/train.py \
    --save_dir ./saved_models/cnn\
    --lattice_size 128 \
    --dataset_size 2048 \
    --epochs 64 \
    --batch_size 32 \
    --dropout_rate 0.0 \
    --learning_rate 1e-3 \
    --device cpu \
    --save_checkpoints
```
## Data augmentation with HYDRA

### Train HYDRA

```shell
python src/hydra/train.py \
    --lattice_size 128 \
    --dataset_size 256 \
    --batch_size 32 \
    --epochs 10 \
    --n_conv_cells 5 \
    --n_convt_cells 5 \
    --generator_learning_rate 1e-5 \
    --discriminator_learning_rate 1e-5 \
    --regularization_strength 1.0 \
    --hydra_ratio_bce 1.0 \
    --hydra_ratio_cnn 0.2 \
    --patience_generator 2 \
    --noise_dim 100 \
    --wanted_p 0.5928 \
    --save_dir ./saved_models/hydra \
    --CNN_model_path ./saved_models/cnn_regression/2022.02.11.15.36.56/model/final_model.pt
``` 
### Generate configurations with a trained HYDRA

```shell
python src/hydra/generate.py \
    --number_images 10 \
    --data_dir ./data/generated \
    --model_dir ./saved_models/hydra/2021.10.17.18.32.10/model/final_model.pt \
    --noise_dim 100
```

## Data augmentation with Convolutional Variational Auto-Encoder (ConvVAE)


### Train ConvVAE

```shell
python src/conv_vae/train.py \
    --dataset_size 512 \
    --save_dir ./saved_models/conv_vae \
    --epochs 64 \
    --learning_rate 1e-3 \
    --batch_size 16 \
    --lattice_size 32 \
    --reg_ratio 1.0 \
    --kl_ratio 1.0 \
    --hidden_dim 512 \
    --latent_dim 16 \
    --device cpu \
    --no-use_property \
    --save_checkpoints
``` 

### Generate configurations with a trained ConvVAE

```shell
python src/conv_vae/generate.py \
    --n_images_per_p 8 \
    --properties 0.5928 \
    --data_dir ./data/conv_vae_generated \
    --model_dir ./saved_models/conv_vae/2022.02.17.13.26.37/Convolutional_VAE_model/final_model.pt
```

## License
[Apache License 2.0](https://github.com/bisonai/mobilenetv3-tensorflow/blob/master/LICENSE)
