import tensorflow as tf
from tensorflow import keras
from transformer import TransformerDecoder
import tensorflow_probability as tfp

class TFIng(keras.Model):
    def __init__(self, crop_size, embed_dim, num_layers, seq_length, hidden_dim, num_heads, 
                    target_vocab_size, dropout_rate=0.1):
        super().__init__()
        self.target_vocab_size = target_vocab_size

        self.encoder = keras.applications.InceptionV3(
            include_top=False, 
            weights="imagenet", 
            input_shape=crop_size + (3,),
        )
        self.conv = keras.layers.Conv2D(embed_dim, 1)
        self.decoder = TransformerDecoder(num_layers, seq_length, embed_dim, hidden_dim, num_heads, 
                                            target_vocab_size, dropout_rate=dropout_rate)
        self.linear = keras.layers.Dense(target_vocab_size, activation=None)

    def call(self, inputs, training=False):
        encoder_inputs, targets = inputs
        encoder_out = self.encoder(encoder_inputs, training=training)
        encoder_out = self.conv(encoder_out, training=training)
        encoder_out = tf.reshape(encoder_out, (tf.shape(encoder_out)[0], -1, tf.shape(encoder_out)[3]))
        decoder_outputs = self.decoder(targets, encoder_out, training=training)
        output = self.linear(decoder_outputs)
        return output + self.get_replacement_mask(targets)

    def get_replacement_mask(self, targets):
        targets = tf.cast(targets, tf.int32)
        batch_size, seq_length = tf.shape(targets)[0], tf.shape(targets)[1]

        n = int(seq_length * (seq_length  + 1) / 2)
        mask = tfp.math.fill_triangular(tf.ones((n,), dtype=tf.int32))
        mask = tf.repeat(mask[tf.newaxis, :], batch_size, axis=0)

        targets_repeated = tf.repeat(targets[:, tf.newaxis, :], seq_length, axis=1)
        targets_masked = targets_repeated * mask
        columns = tf.boolean_mask(
            targets_masked, 
            tf.where(targets_masked != 0, tf.ones_like(targets_masked), tf.zeros_like(targets_masked))
        )

        rows_idx = tf.range(seq_length)
        rows_idx_repeated = tf.reshape(tf.repeat(rows_idx, seq_length), (seq_length, seq_length))
        rows_idx_repeated = tf.repeat(rows_idx_repeated[tf.newaxis, :], batch_size, axis=0)
        rows = tf.boolean_mask(
            rows_idx_repeated, 
            tf.where(targets_masked != 0, tf.ones_like(targets_masked), tf.zeros_like(targets_masked))
        )

        batches_idx = tf.range(batch_size)
        batches_idx_repeated = tf.reshape(
            tf.repeat(batches_idx, seq_length * seq_length), (batch_size, seq_length, seq_length)
        )
        batches = tf.boolean_mask(
            batches_idx_repeated, 
            tf.where(targets_masked != 0, tf.ones_like(targets_masked), tf.zeros_like(targets_masked))
        )

        idx = tf.stack([batches, rows, columns], axis=1)

        sparse_mask = tf.SparseTensor(
            tf.cast(idx, tf.int64), 
            tf.fill([tf.shape(idx)[0]], float('-inf')), 
            [batch_size, seq_length, self.target_vocab_size]
        )
        sparse_mask = tf.sparse.reorder(sparse_mask)
        return tf.sparse.to_dense(sparse_mask)
