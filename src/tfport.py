import tensorflow as tf
from tensorflow import keras
from transformer import TransformerEncoder, TransformerDecoderLayer, get_look_ahead_mask, get_padding_mask

class TFPort(keras.Model):
    def __init__(self, crop_size, embed_dim, encoder_layers, decoder_layers, input_seq_length, 
                    target_seq_length, hidden_dim, num_heads, input_vocab_size, dropout_rate=0.1):
        super().__init__()

        self.image_encoder = keras.applications.InceptionV3(
            include_top=False, 
            weights='inception_w.h5',
            input_shape=crop_size + (3,),
        )
        self.image_encoder.trainable = False
        self.conv = keras.layers.Conv2D(embed_dim, 1)
        self.ingredient_encoder = TransformerEncoder(encoder_layers, input_seq_length, embed_dim, hidden_dim, 
                                                        num_heads, input_vocab_size, dropout_rate=dropout_rate)
        self.portion_embedding = PortionEmbedding(target_seq_length, embed_dim)
        self.dropout = keras.layers.Dropout(dropout_rate)
        self.decoder_layers = [
            TransformerDecoderLayer(embed_dim, hidden_dim, num_heads, dropout_rate=dropout_rate)
            for _ in range(decoder_layers)
        ]
        self.linear = keras.layers.Dense(1, activation="relu")

    def call(self, inputs, training=False):
        image, ingredients, targets = inputs
        padding_mask = get_padding_mask(ingredients)
        encoded_img = self.image_encoder(image, training=False)
        encoded_img = self.conv(encoded_img, training=training)
        encoded_img = tf.reshape(encoded_img, (tf.shape(encoded_img)[0], -1, tf.shape(encoded_img)[3]))
        encoded_ingr = self.ingredient_encoder(ingredients, padding_mask, training=training)
        encoder_outputs = tf.concat([encoded_img, encoded_ingr], axis=1)

        img_mask = tf.ones((tf.shape(encoded_img)[0], 1, tf.shape(encoded_img)[1]), dtype=tf.int32)
        padding_mask = tf.concat([img_mask, padding_mask], axis=2)
        look_ahead_mask = get_look_ahead_mask(targets)
        
        x = self.portion_embedding(targets)
        x = self.dropout(x, training=training)
        for i in range(len(self.decoder_layers)):
            x = self.decoder_layers[i](x, encoder_outputs, look_ahead_mask, training=training, 
                                        padding_mask=padding_mask)
        x = self.linear(x)
        return tf.squeeze(x)


class PortionEmbedding(keras.layers.Layer):
    def __init__(self, seq_length, embed_dim):
        super().__init__()

        self.linear = keras.layers.Dense(embed_dim)
        self.position_embeddings = keras.layers.Embedding(
            input_dim=seq_length, output_dim=embed_dim
        )

    def call(self, inputs):
        positions = tf.range(start=0, limit=tf.shape(inputs)[-1], delta=1)
        embedded_portions = self.linear(inputs[:, :, tf.newaxis])
        embedded_positions = self.position_embeddings(positions)
        return embedded_portions + embedded_positions
