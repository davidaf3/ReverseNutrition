import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras

class TransformerEncoderLayer(keras.layers.Layer):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout_rate=0.1):
        super().__init__()

        self.attention = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.feed_forward = keras.Sequential(
            [
                keras.layers.Dense(hidden_dim, activation="relu"), 
                keras.layers.Dense(embed_dim, activation=None)
            ]
        )

        self.layernorm1 = keras.layers.LayerNormalization()
        self.layernorm2 = keras.layers.LayerNormalization()
        
        self.dropout1 = keras.layers.Dropout(dropout_rate)
        self.dropout2 = keras.layers.Dropout(dropout_rate)

    def call(self, inputs, padding_mask, training=False):
        attn_out = self.attention(
            query=inputs, 
            value=inputs, 
            key=inputs, 
            attention_mask=padding_mask
        )
        attn_out = self.dropout1(attn_out, training=training)
        x = self.layernorm1(inputs + attn_out)

        ff_out = self.feed_forward(x)
        ff_out = self.dropout2(ff_out, training=training)
        return self.layernorm2(x + ff_out)


class TransformerEncoder(keras.Model):
    def __init__(self, num_layers, seq_length, embed_dim, hidden_dim, num_heads, vocab_size, 
                dropout_rate=0.1):
        super().__init__()

        self.embedding = PositionalEmbedding(seq_length, vocab_size, embed_dim)
        self.dropout = keras.layers.Dropout(dropout_rate)
        self.encoder_layers = [
            TransformerEncoderLayer(embed_dim, hidden_dim, num_heads, dropout_rate=dropout_rate)
            for _ in range(num_layers)
        ]

    def call(self, inputs, padding_mask, training=False):
        x = self.embedding(inputs)
        x = self.dropout(x, training=training)
        for i in range(len(self.encoder_layers)):
            x = self.encoder_layers[i](x, padding_mask, training=training)
        return x


class TransformerDecoderLayer(keras.layers.Layer):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout_rate=0.1):
        super().__init__()

        self.self_attention = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.attention = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.feed_fordward = keras.Sequential(
            [
                keras.layers.Dense(hidden_dim, activation="relu"), 
                keras.layers.Dense(embed_dim, activation=None)
            ]
        )
        
        self.layernorm1 = keras.layers.LayerNormalization()
        self.layernorm2 = keras.layers.LayerNormalization()
        self.layernorm3 = keras.layers.LayerNormalization()

        self.dropout1 = keras.layers.Dropout(dropout_rate)
        self.dropout2 = keras.layers.Dropout(dropout_rate)
        self.dropout3 = keras.layers.Dropout(dropout_rate)

    def call(self, inputs, encoder_outputs, look_ahead_mask, training=False, padding_mask=None):
        self_attn_out = self.self_attention(
            query=inputs, 
            value=inputs, 
            key=inputs, 
            attention_mask=look_ahead_mask
        )
        self_attn_out = self.dropout1(self_attn_out, training=training)
        x = self.layernorm1(inputs + self_attn_out)

        attn_out = self.attention(
            query=x,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask
        )
        attn_out = self.dropout2(attn_out, training=training)
        x = self.layernorm2(x + attn_out)

        ff_out = self.feed_fordward(x)
        ff_out = self.dropout3(ff_out, training=training)
        return self.layernorm3(x + ff_out)


class TransformerDecoder(keras.Model):
    def __init__(self, num_layers, seq_length, embed_dim, hidden_dim, num_heads, vocab_size, 
                dropout_rate=0.1):
        super().__init__()

        self.embedding = PositionalEmbedding(seq_length, vocab_size, embed_dim)
        self.dropout = keras.layers.Dropout(dropout_rate)
        self.decoder_layers = [
            TransformerDecoderLayer(embed_dim, hidden_dim, num_heads, dropout_rate=dropout_rate)
            for _ in range(num_layers)
        ]

    def call(self, inputs, encoder_outputs, training=False, padding_mask=None):
        look_ahead_mask = get_look_ahead_mask(inputs)
        x = self.embedding(inputs)
        x = self.dropout(x, training=training)
        for i in range(len(self.decoder_layers)):
            x = self.decoder_layers[i](x, encoder_outputs, look_ahead_mask, 
                                        training=training, padding_mask=padding_mask)
        return x


def get_padding_mask(inputs):
    mask = tf.cast(tf.math.not_equal(inputs, 0), tf.int32)
    return mask[:, tf.newaxis, :]

def get_look_ahead_mask(inputs):
    input_shape = tf.shape(inputs)
    batch_size, seq_length = input_shape[0], input_shape[1]
    n = int(seq_length * (seq_length  + 1) / 2)
    mask = tfp.math.fill_triangular(tf.ones((n,), dtype=tf.int32))
    mask = tf.repeat(mask[tf.newaxis, :], batch_size, axis=0)
    return tf.minimum(mask, get_padding_mask(inputs))


class PositionalEmbedding(keras.layers.Layer):
    def __init__(self, seq_length, vocab_size, embed_dim):
        super().__init__()

        self.token_embeddings = keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = keras.layers.Embedding(
            input_dim=seq_length, output_dim=embed_dim
        )

    def call(self, inputs):
        positions = tf.range(start=0, limit=tf.shape(inputs)[-1], delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions


class Transformer(keras.Model):
    def __init__(self, encoder_layers, decoder_layers, input_seq_length, target_seq_length, embed_dim, 
                    hidden_dim, num_heads, input_vocab_size, target_vocab_size, dropout_rate=0.1):
        super().__init__()
        
        self.encoder = TransformerEncoder(encoder_layers, input_seq_length, embed_dim, hidden_dim, 
                                            num_heads, input_vocab_size, dropout_rate=dropout_rate)
        self.decoder = TransformerDecoder(decoder_layers, target_seq_length, embed_dim, hidden_dim, 
                                            num_heads, target_vocab_size, dropout_rate=dropout_rate)
        self.linear = keras.layers.Dense(target_vocab_size, activation=None)

    def call(self, inputs, training=False):
        encoder_inputs, targets = inputs
        padding_mask = get_padding_mask(encoder_inputs)
        encoder_outputs = self.encoder(encoder_inputs, padding_mask, training=training)
        decoder_outputs = self.decoder(targets, encoder_outputs, training=training, 
                                        padding_mask=padding_mask)
        return self.linear(decoder_outputs)