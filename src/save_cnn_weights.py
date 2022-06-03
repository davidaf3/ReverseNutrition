import tensorflow as tf
from tensorflow import keras
from tfing import TFIng
from train_tfing import make_loss, accuracy_function

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

model_name = '../models/tfing'
dropout_rate = 0.1

model = TFIng(crop_size, embed_dim, num_layers, seq_length, hidden_dim, num_heads, 
                            558, dropout_rate=dropout_rate)
model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate), 
        loss=make_loss(), 
        metrics=[accuracy_function]
)
model((tf.zeros((1, 224, 224, 3)), tf.zeros((1, seq_length))))
model.load_weights(f'{model_name}.h5')

model.encoder.save_weights('inception_w.h5')
