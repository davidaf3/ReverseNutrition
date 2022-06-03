import json
from tensorflow import keras
from data_loader import make_dataset
from fcnutr import FCNutr

nutr_names = ('energy', 'fat', 'protein', 'carbs')
nutrition5k_path = ''
batch_size = 32
img_size = 254
crop_size = (224, 224)
hidden_dim = 4096
learning_rate = 1e-4
epochs = 80
patience = 10

model_name = 'fcnutr_3-shared_no-dropout'
num_shared_hidden = 3
use_dropout = False

def main():
    with open(f'{nutrition5k_path}/metadata/dish_metadata_cafe1.json', encoding='UTF-8') as f:
        dishes = json.load(f)

    train_ds, val_ds = tuple(
        make_dataset(
            partition,
            nutrition5k_path,
            dishes,
            nutr_names,
            keras.applications.inception_v3.preprocess_input,
            crop_size,
            img_size,
            batch_size
        ) for partition in ('train', 'val')
    )

    model = FCNutr(nutr_names, crop_size, hidden_dim, num_shared_hidden, use_dropout)
    model.compile(
         optimizer=keras.optimizers.Adam(learning_rate=learning_rate), 
         loss=make_mae(nutr_names), 
         metrics=make_mse(nutr_names)
    )

    callbacks = [
        keras.callbacks.BackupAndRestore(backup_dir='./backup'),
        keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1),
        keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)
    ]

    model.fit(train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds)
    model.save(model_name, save_format="tf")

def make_mae(nutr_names):
    mae = keras.losses.MeanAbsoluteError()

    def mae_of(i):
        fn = lambda labels, prediction: mae(labels[:, i], prediction)
        fn.__name__ = 'mae'
        return fn

    return {name: mae_of(i) for i, name in enumerate(nutr_names)}

def make_mse(nutr_names):
    mse = keras.losses.MeanSquaredError()

    def mse_of(i):
        fn = lambda labels, prediction: mse(labels[:, i], prediction)
        fn.__name__ = 'mse'
        return fn

    return {name: mse_of(i) for i, name in enumerate(nutr_names)}


if __name__ == '__main__':
    main()