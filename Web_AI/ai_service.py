import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import json
import time
from django.conf import settings
from Web_AI.chatbot_model import Seq2SeqModel, Encoder, Decoder
import numpy as np
import tensorflow as tf

# ...existing code...
# ========================================================== 
# PHẦN TẢI MÔ HÌNH THỰC TẾ
# ==========================================================

# Load config
with open(os.path.join(settings.BASE_DIR, 'ai_model_chats', 'config.json')) as f:
    config = json.load(f)

# Load tokenizers
with open(os.path.join(settings.BASE_DIR, 'ai_model_chats', 'cauhoi_tokenizer.pkl'), 'rb') as f:
    tokenizer_input = pickle.load(f)

with open(os.path.join(settings.BASE_DIR, 'ai_model_chats', 'traloi_tokenizer.pkl'), 'rb') as f:
    tokenizer_output = pickle.load(f)


def create_positional_embedding(max_len, model_size):
    position = np.arange(max_len)[:, np.newaxis]
    i = np.arange(model_size)[np.newaxis, :]
    angle_rates = position / np.power(10000, (2 * (i // 2)) / model_size)
    angle_rates[:, 0::2] = np.sin(angle_rates[:, 0::2])
    angle_rates[:, 1::2] = np.cos(angle_rates[:, 1::2])
    return tf.constant(angle_rates, dtype=tf.float32)

pes = create_positional_embedding(
    max_len=max(config["MAX_LEN_CAUHOI"], config["MAX_LEN_TRALOI"]),
    model_size=config["MODEL_SIZE"]
)

encoder = Encoder(
    vocab_size=config["VOCAB_SIZE_INPUT"],
    model_size=config["MODEL_SIZE"],
    num_layers=config["NUM_LAYERS"],
    h=config["NUM_HEADS"]
)

decoder = Decoder(
    vocab_size=config["VOCAB_SIZE_OUTPUT"],
    model_size=config["MODEL_SIZE"],
    num_layers=config["NUM_LAYERS"],
    h=config["NUM_HEADS"]
)

model = Seq2SeqModel(encoder, decoder)
dummy_input = tf.zeros((1, config["MAX_LEN_CAUHOI"]), dtype=tf.int32)
dummy_decoder = tf.zeros((1, 1), dtype=tf.int32)
_ = model(dummy_input, dummy_decoder)

encoder.load_weights(os.path.join(settings.BASE_DIR, 'ai_model_chats', 'encoder_weights.weights.h5'))
decoder.load_weights(os.path.join(settings.BASE_DIR, 'ai_model_chats', 'decoder_weights.weights.h5'))
# ==========================================================
# PHẦN DỰ ĐOÁN TRẢ LỜI
# ==========================================================

def predict(user_input: str, max_len=40) -> str:
    seq = tokenizer_input.texts_to_sequences([user_input])
    padded_input = tf.keras.preprocessing.sequence.pad_sequences(
        seq, maxlen=config["MAX_LEN_CAUHOI"], padding='post'
    )

    decoder_input = tf.constant([[tokenizer_output.word_index.get('<start>', 1)]], dtype=tf.int64)
    output_words = []

    for _ in range(max_len):
        predictions = model(padded_input, decoder_input)
        next_token = tf.argmax(predictions[:, -1, :], axis=-1).numpy()[0]
        next_word = tokenizer_output.index_word.get(next_token, '')

        if next_word == '<end>' or next_word == '':
            break

        output_words.append(next_word)
        decoder_input = tf.concat([decoder_input, [[next_token]]], axis=-1)

    return ' '.join(output_words) if output_words else "Xin lỗi, tôi chưa có câu trả lời phù hợp."


