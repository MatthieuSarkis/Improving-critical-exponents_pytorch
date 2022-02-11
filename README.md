# Improving critical exponent estimations with Generative Adversarial Networks

## Requirements

* Python 3.8+
* torch
* sklearn
* scipy
* tqdm
* jupyter
* pandas

```shell
pip install .
```
## Control parameter estimation via CNN

It possible to train a CNN with the following command:

 ```shell
python src/cnn/train.py \
    --dataset_size 1024 \
    --lattice_size 128 \
    --batch_size 64 \
    --epochs 40 \
    --learning_rate 1e-4 
 ```

 The trained CNN is used to as one of the two heads in Hydra.

## Data augmentation via GAN

### Train generative model

```shell
python src/hydra/train.py \
    --lattice_size 128 \
    --dataset_size 256 \
    --batch_size 32 \
    --epochs 10 \
    --n_conv_cells 2 \
    --n_convt_cells 5 \
    --generator_learning_rate 1e-3 \
    --discriminator_learning_rate 1e-3 \
    --regularization_strength 1.0 \
    --hydra_ratio_bce 1.0 \
    --hydra_ratio_cnn 1.0 \
    --patience_generator 2 \
    --noise_dim 100 \
    --wanted_p 0.5928 \
    --save_dir ./saved_models/hydra \
    --CNN_model_path ./saved_models/cnn_regression/2022.02.11.15.36.56/model/final_model.pt
``` 

### Generate configurations with GAN

```shell
python src/hydra/generate.py \
    --number_images 10 \
    --data_dir ./data/generated \
    --model_dir ./saved_models/hydra/2021.10.17.18.32.10/model/final_model.pt \
    --noise_dim 100
```

## License
[Apache License 2.0](https://github.com/bisonai/mobilenetv3-tensorflow/blob/master/LICENSE)
