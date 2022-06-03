import json
import tensorflow as tf
from tensorflow import keras
from data_loader import make_ingredients_dataset
from tfing import TFIng

nutr_names = ('energy', 'fat', 'protein', 'carbs')
nutrition5k_path = ''
batch_size = 32
img_size = 256
crop_size = (224, 224)
seq_length = 20
num_layers = 3
num_heads = 8
embed_dim = 256
hidden_dim = 1024
learning_rate = 1e-5
epochs = 80
patience = 10

model_name = 'tfing'
dropout_rate = 0.1

def main():
    with open(f'{nutrition5k_path}/metadata/dish_metadata_cafe1_sorted.json', encoding='UTF-8') as f:
        dishes_sorted = json.load(f)

    with open(f'{nutrition5k_path}/metadata/ingredients_metadata.json', encoding='UTF-8') as f:
        vocab_size = len(json.load(f).keys()) + 3

    train_ds, val_ds = tuple(
        make_ingredients_dataset(
            partition,
            seq_length,
            vocab_size,
            nutrition5k_path,
            dishes_sorted,
            keras.applications.inception_v3.preprocess_input,
            crop_size,
            img_size,
            batch_size
        ) for partition in ('train', 'val')
    )

    model = TFIng(crop_size, embed_dim, num_layers, seq_length, hidden_dim, num_heads, 
                            vocab_size, dropout_rate=dropout_rate)
    model.compile(
         optimizer=keras.optimizers.Adam(learning_rate=learning_rate), 
         loss=make_loss(), 
         metrics=[accuracy_function]
    )

    callbacks = [
        keras.callbacks.BackupAndRestore(backup_dir='./backup'),
        keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1),
        keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)
    ]

    model.fit(train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds)
    model.save(model_name, save_format="tf")

def make_loss():
    loss_object = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none'
    )

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

    return loss_function

def accuracy_function(real, pred):
    real = tf.cast(real, dtype=tf.int64)
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


if __name__ == '__main__':
    main()