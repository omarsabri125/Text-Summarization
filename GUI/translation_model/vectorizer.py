import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import pandas as pd
from .config import CONFIG
import sys
import io


english_vectorizer = tf.keras.layers.TextVectorization(
    max_tokens= CONFIG["VOCAB_SIZE"],
    output_mode ='int',
    output_sequence_length = CONFIG["SEQUENCE_LENGTH"]
)

arabic_vectorizer = tf.keras.layers.TextVectorization(
    max_tokens = CONFIG["VOCAB_SIZE"],
    output_mode ='int',
    output_sequence_length = CONFIG["SEQUENCE_LENGTH"] + 1
)

# Load the English and Arabic text files

# Force UTF-8 for stdout (for Windows)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
train_df = pd.read_csv(CONFIG["train_data_path"])

english_vectorizer.adapt(train_df['english'].values)
arabic_vectorizer.adapt(train_df['arabic'].values)