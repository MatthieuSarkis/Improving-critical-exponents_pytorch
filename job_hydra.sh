python src/hydra/train.py \
    --lattice_size 128 \
    --dataset_size 50 \
    --batch_size 10 \
    --epochs 2 \
    --bins_number 100 \
    --learning_rate 10e-3 \
    --regularization_strength 1.0 \
    --discriminator_cnn_ratio 1.0 \
    --noise_dim 100 \
    --wanted_p 0.5928 \
    --save_dir "./saved_models/hydra" \
    --CNN_model_path "./saved_models/cnn_regression/2021.10.19.11.09.07/model/final_model.pt" \
    --device "cpu" \
    --no-set_generate_plots