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
python setup.py install
```
## Control parameter estimation via CNN

It possible to train a CNN with the following command:

 ```shell
python src/CNN_regression/train.py \
    --dataset_size 1024 \
    --lattice_size 128 \
    --batch_size 64 \
    --epochs 10 \
    --learning_rate 1e-4 
 ```

 The trained CNN is used to as one of the two heads in Hydra.

## Data augmentation via GAN

### Train generative model

```shell
python src/GAN_CNNRegression/train.py \
	--epochs 2 \
	--batch_size 3 \
	--noise_dim 100 \
	--CNN_model_path ./saved_models/cnn_regression/2021.10.17.00.35.47/model/final_model.pt \
	--bins_number 100 \
	--no-set_generate_plots 
``` 

### Generate configurations with GAN

```shell
python src/GAN/generate.py \
    --num 10 \
    --data_dir ./data/generated/ \
    --model_dir ./data/models/gan/ \
    --noise_dim 100
```

## License
[Apache License 2.0](https://github.com/bisonai/mobilenetv3-tensorflow/blob/master/LICENSE)
