# Copyright (C) 2024 - 2024 Ruojian Li， Inc. All Rights Reserved
# @Time    : 2024/3/8 21:30
# @Email   : ruojian_li@163.com
# License: MIT
from pypots.optim import Adam
from pypots.utils.metrics import cal_mae
from pypots.utils.random import set_random_seed
from Global_Config import RANDOM_SEED
from data.gene_highd import gene_highd
from module.MAITTS import MAITTS

if __name__ == '__main__':
    set_random_seed(RANDOM_SEED)

    MISIING_RATE = 0.3
    dataset_path = "data_preprocessing/experimental_data/fixed_detector_floating_car_scene_data_" + str(MISIING_RATE)
    highd_dataset = gene_highd(dataset_path, artificially_missing_rate=0.1)

    dataset_for_training = {
        "X": highd_dataset['train_X'],
    }

    dataset_for_validating = {
        "X": highd_dataset['val_X'],
        "X_intact": highd_dataset['val_X_intact'],
        "indicating_mask": highd_dataset['val_X_indicating_mask'],
    }

    dataset_for_testing = {
        "X": highd_dataset['test_X'],
    }

    model = MAITTS(
        n_steps=highd_dataset['n_steps'],
        n_features=highd_dataset['n_features'],
        n_layers=8,
        n_mixer_layers=4,
        d_model=128, d_inner=256, n_heads=6,
        d_k=32, d_v=32, dropout=0, attn_dropout=0.1,
        diagonal_attention_mask=True,
        ORT_weight=1, MIT_weight=1,
        batch_size=128, epochs=10000, patience=50,
        optimizer=Adam(lr=0.0001),
        num_workers=4,
        device='cuda:0',
        saving_path="tutorial_results/highd_imputation/MAPITS",
        model_saving_strategy="best",
    )
    model.fit(train_set=dataset_for_training, val_set=dataset_for_validating)

    model_imputation = model.impute(dataset_for_testing)
    testing_mae = cal_mae(
        model_imputation, highd_dataset['test_X_intact'], highd_dataset['test_X_indicating_mask'])
    print(testing_mae)
