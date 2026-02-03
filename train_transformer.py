import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import os

# --- Configuration ---
MAX_LEN = 200  # Maximum sequence length
VOCAB_SIZE = 100  # Rough estimate for printable characters
EMBED_DIM = 32
NUM_HEADS = 2
FF_DIM = 32
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001

def load_dataset(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} not found.")
    df = pd.read_csv(filename)
    x = df['url'].values
    y = df['label'].values
    return x, y

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super().get_config()
        config.update({
            "maxlen": self.maxlen,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
        })
        return config

def main():
    print("Loading data...")
    x_train, y_train = load_dataset("train.csv")
    x_val, y_val = load_dataset("val.csv")
    x_test, y_test = load_dataset("test.csv")

    print(f"Train samples: {len(x_train)}")
    print(f"Val samples: {len(x_val)}")
    print(f"Test samples: {len(x_test)}")

    # Vectorization
    print("Vectorizing data...")
    vectorizer = layers.TextVectorization(
        max_tokens=None, # Allow full vocabulary
        output_mode="int",
        output_sequence_length=MAX_LEN,
        split="character", # Character-level tokenization
        standardize=None # No standardization (raw URLs)
    )
    vectorizer.adapt(x_train)
    
    vocab = vectorizer.get_vocabulary()
    real_vocab_size = len(vocab)
    print(f"Vocabulary size: {real_vocab_size}")
    
    # Define Model
    inputs = layers.Input(shape=(1,), dtype=tf.string)
    x = vectorizer(inputs)
    embedding_layer = TokenAndPositionEmbedding(MAX_LEN, real_vocab_size, EMBED_DIM)
    x = embedding_layer(x)
    transformer_block = TransformerBlock(EMBED_DIM, NUM_HEADS, FF_DIM)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")]
    )
    
    model.summary()

    print("Starting training...")
    history = model.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_val, y_val)
    )

    print("\nEvaluating on Test Set...")
    results = model.evaluate(x_test, y_test)
    print("Test Loss, Test Acc, Test Precision, Test Recall:", results)

    # Save model
    model.save("transformer_phishing_model_v2.keras")
    print("Model saved to transformer_phishing_model_v2.keras")

if __name__ == "__main__":
    main()
