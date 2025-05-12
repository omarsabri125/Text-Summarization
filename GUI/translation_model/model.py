import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers # type: ignore
import keras
from .config import CONFIG


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim)

        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim)

        self.sequence_length = sequence_length
        self.vocab_size = vocab_size # input dimm
        self.embed_dim = embed_dim # output dimm

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return keras.ops.not_equal(inputs, 0)

    def get_config(self):
        config = super(PositionalEmbedding, self).get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
        })
        return config

def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def attention_mask(nd, ns, *, dtype):
    """1's in the lower triangle, counting from the lower right corner.
    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    i = tf.range(nd)[:,None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)

def mask_attn_weights(w):
    # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
    _, _, nd, ns = shape_list(w)
    b = attention_mask(nd, ns, dtype=w.dtype)
    b = tf.reshape(b, [1, 1, nd, ns])
    w = w*b - tf.cast(1e10, w.dtype)*(1-b)
    return w

def scaled_dot_product_attention(q, k, v, mask=False):
  d_k = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(d_k)
  if mask:
    scaled_attention_logits = mask_attn_weights(scaled_attention_logits)
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
  output = tf.matmul(attention_weights, v)
  return output

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def concat_heads(self, x, batch_size):
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return tf.reshape(x, (batch_size, -1, self.d_model))

    def call(self, q, v, k,mask=False):
        batch_size = tf.shape(k)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_attention = scaled_dot_product_attention(q, k, v,mask)
        concat_attention = self.concat_heads(scaled_attention, batch_size)
        output = self.dense(concat_attention)
        return output

    def get_config(self):
        config = super(MultiHeadAttention, self).get_config()
        config.update({
            "num_heads": self.num_heads,
            "d_model": self.d_model,
        })
        return config
    
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x):
        attn_output = self.mha(x, x, x)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

    def get_config(self):
        config = super(EncoderLayer, self).get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "rate": self.rate,
        })
        return config

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output):
        attn1 = self.mha1(x, x, x, mask = True)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(attn1 + x)
        attn2 = self.mha2(out1, enc_output, enc_output)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(attn2 + out1)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(ffn_output + out2)
        return out3
    
def build_tranformer_model():
    #build transformer
    encoder_inputs = tf.keras.Input(shape=(None,), dtype="int64", name="source")
    x = PositionalEmbedding(CONFIG["SEQUENCE_LENGTH"], CONFIG["VOCAB_SIZE"], CONFIG["EMBEDDING_DIM"])(encoder_inputs)
    encoder_outputs = EncoderLayer(CONFIG["EMBEDDING_DIM"], CONFIG["NUM_HEADS"], CONFIG["UNITS"])(x)
    decoder_inputs = tf.keras.Input(shape=(None,), dtype="int64", name="target")
    x = PositionalEmbedding(CONFIG["SEQUENCE_LENGTH"], CONFIG["VOCAB_SIZE"], CONFIG["EMBEDDING_DIM"])(decoder_inputs)
    x = DecoderLayer(CONFIG["EMBEDDING_DIM"], CONFIG["NUM_HEADS"], CONFIG["UNITS"])(x, encoder_outputs)
    x = layers.Dropout(0.5)(x)
    decoder_outputs = layers.Dense(CONFIG["VOCAB_SIZE"], activation="softmax")(x)
    transformer = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return transformer
