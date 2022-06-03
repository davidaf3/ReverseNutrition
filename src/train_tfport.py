import json
import tensorflow as tf
from tensorflow import keras
from data_loader import make_portions_dataset
from tfport import TFPort

nutr_names = ('energy', 'fat', 'protein', 'carbs')
nutrition5k_path = ''
batch_size = 32
img_size = 256
crop_size = (224, 224)
seq_length = 20
num_encoder_layers = 3
num_decoder_layers = 3
num_heads = 8
embed_dim = 256
hidden_dim = 1024
learning_rate = 1e-5
epochs = 80
patience = 10

model_name = 'tfport'
dropout_rate = 0.1

def main():
    with open(f'{nutrition5k_path}/metadata/dish_metadata_cafe1_2+ing.json', encoding='UTF-8') as f:
        dishes_2plusing = json.load(f)

    with open(f'{nutrition5k_path}/metadata/ingredients_metadata.json', encoding='UTF-8') as f:
        vocab_size = len(json.load(f).keys()) + 3

    train_ds, val_ds = tuple(
        make_portions_dataset(
            partition,
            seq_length,
            vocab_size,
            nutrition5k_path,
            dishes_2plusing,
            keras.applications.inception_v3.preprocess_input,
            crop_size,
            img_size,
            batch_size
        ) for partition in ('train', 'val')
    )

    model = TFPort(crop_size, embed_dim, num_encoder_layers, num_decoder_layers, seq_length, seq_length, 
                            hidden_dim, num_heads, vocab_size, dropout_rate=dropout_rate)
    model.compile(
         optimizer=keras.optimizers.Adam(learning_rate=learning_rate), 
         loss=mae, 
         metrics=[mse]
    )

    callbacks = [
        keras.callbacks.BackupAndRestore(backup_dir='./backup'),
        keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1),
        keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)
    ]

    model.fit(train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds)
    model.save(model_name, save_format="tf")

def mae(real, pred):
    abserr = tf.abs(real - pred)
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    mask = tf.cast(mask, dtype=abserr.dtype)
    abserr *= mask
    return tf.reduce_mean(tf.reduce_sum(abserr, axis=1) / tf.reduce_sum(mask, axis=1))

def mse(real, pred):
    sqerr = tf.pow(real - pred, 2)
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    mask = tf.cast(mask, dtype=sqerr.dtype)
    sqerr *= mask
    return tf.reduce_mean(tf.reduce_sum(sqerr, axis=1) / tf.reduce_sum(mask, axis=1))

if __name__ == '__main__':
    main()