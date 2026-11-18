import tensorflow as tf

class GPTConfig:
    def __init__(
        self,
        vocab_size,
        max_length=512,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        dropout=0.1,
        pad_id=0,
    ):
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.vocab_size = int(vocab_size)
        self.max_length = int(max_length)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.d_ff = int(d_ff)
        self.dropout = float(dropout)
        self.pad_id = int(pad_id)

    def to_dict(self):
        return dict(
            vocab_size=self.vocab_size, max_length=self.max_length,
            d_model=self.d_model, num_heads=self.num_heads,
            num_layers=self.num_layers, d_ff=self.d_ff,
            dropout=self.dropout, pad_id=self.pad_id
        )


class CausalSelfAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads.")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = tf.keras.layers.Dense(3 * d_model, use_bias=True)
        self.proj = tf.keras.layers.Dense(d_model)
        self.attn_dropout = tf.keras.layers.Dropout(dropout)
        self.resid_dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, token_mask=None, training=False):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        qkv = self.qkv(x)
        q, k, v = tf.split(qkv, 3, axis=-1)

        def split_heads(t):
            t = tf.reshape(t, [batch_size, seq_len, self.num_heads, self.head_dim])
            return tf.transpose(t, [0, 2, 1, 3])

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        scores = tf.matmul(q, k, transpose_b=True)
        scores = scores / tf.math.sqrt(tf.cast(self.head_dim, scores.dtype))

        # Causal mask: a token can only attend to itself and earlier tokens.
        causal = tf.linalg.band_part(tf.ones([seq_len, seq_len], dtype=tf.float32), -1, 0) > 0
        scores = tf.where(causal[None, None, :, :], scores,
                          tf.cast(-1e9, scores.dtype))

        # Mask padding keys so real tokens cannot attend to padding.
        if token_mask is not None:
            key_mask = tf.cast(token_mask[:, None, None, :], tf.bool)
            scores = tf.where(key_mask, scores, tf.cast(-1e9, scores.dtype))

        weights = tf.nn.softmax(scores, axis=-1)
        weights = self.attn_dropout(weights, training=training)
        out = tf.matmul(weights, v)
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [batch_size, seq_len, self.d_model])
        return self.resid_dropout(self.proj(out), training=training)


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(d_ff, activation=tf.nn.gelu),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout),
        ])

    def call(self, x, training=False):
        return self.net(x, training=training)


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.attn = CausalSelfAttention(d_model, num_heads, dropout)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def call(self, x, token_mask=None, training=False):
        x = x + self.attn(self.ln1(x), token_mask=token_mask, training=training)
        x = x + self.ff(self.ln2(x), training=training)
        return x


class GPTSummarizer(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.token_embedding = tf.keras.layers.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = tf.keras.layers.Embedding(config.max_length, config.d_model)
        self.embedding_dropout = tf.keras.layers.Dropout(config.dropout)
        self.blocks = [
            TransformerBlock(config.d_model, config.num_heads, config.d_ff, config.dropout)
            for _ in range(config.num_layers)
        ]
        self.final_ln = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.lm_head = tf.keras.layers.Dense(config.vocab_size, use_bias=False)

    def call(self, input_ids, training=False):
        seq_len = tf.shape(input_ids)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        token_mask = tf.not_equal(input_ids, self.config.pad_id)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        x = self.embedding_dropout(x, training=training)
        for block in self.blocks:
            x = block(x, token_mask=token_mask, training=training)
        x = self.final_ln(x)
        return self.lm_head(x)

    def get_config_dict(self):
        return self.config.to_dict()
