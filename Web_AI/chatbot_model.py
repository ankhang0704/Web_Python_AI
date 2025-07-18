import numpy as np
import tensorflow as tf
import keras
from keras.saving import register_keras_serializable, serialize_keras_object, deserialize_keras_object
from keras.models import Model
import os
from django.conf import settings
import json
import re

pes = []
with open(os.path.join(settings.BASE_DIR, 'ai_model_chats', 'config.json')) as f:
    config = json.load(f)


data = {    'cauhoi': [
        "Bạn tên gì?",
        "Bạn bao nhiêu tuổi?",
        "Bạn đến từ đâu?",
        "Bạn thích làm gì?",
        "Bạn có sở thích gì không?"
    ],
    'traloi': [
        "Tôi là một chatbot.",
        "Tôi không có tuổi.",
        "Tôi đến từ thế giới ảo.",
        "Tôi thích trò chuyện với mọi người.",
        "Tôi thích học hỏi và cải thiện."
    ]
}

#!pip install pyvi
#from pyvi import ViTokenizer
##### Câu hỏi
data_cauhoi = []
for tach1 in data['cauhoi']:
        #tachtu1 = ViTokenizer.tokenize(tach1)
        #data_cauhoi.append(tachtu1)
        data_cauhoi.append(tach1)
#print(data_cauhoi)
##### Trả lời
data_traloi = []
for tach2 in data['traloi']:
        #tachtu2 = ViTokenizer.tokenize(tach2)
        #data_traloi.append(tachtu2)
        data_traloi.append(tach2)
#print(data_traloi)


# Hàm chuyển chữ thường
def normalize_string(text):
    text = re.sub(r"\'ll", " \'ll", text)
    return text.strip().lower()

#raw_data_en, raw_data_fr = list(zip(*raw_data))
#raw_data_en, raw_data_fr = list(raw_data_en), list(raw_data_fr)
raw_data_cauhoi = [normalize_string(data) for data in data_cauhoi]
raw_data_traloi_in = ['<start> ' + normalize_string(data) for data in data_traloi]
raw_data_traloi_out = [normalize_string(data) + ' <end>' for data in data_traloi]


MAX_LEN_CAUHOI = 60
MAX_LEN_TRALOI = 120

cauhoi_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
cauhoi_tokenizer.fit_on_texts(raw_data_cauhoi)
data_cauhoi = cauhoi_tokenizer.texts_to_sequences(raw_data_cauhoi)
data_cauhoi = tf.keras.preprocessing.sequence.pad_sequences(data_cauhoi,
                                                        padding='post',maxlen=MAX_LEN_CAUHOI)

traloi_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
traloi_tokenizer.fit_on_texts(raw_data_traloi_in)
traloi_tokenizer.fit_on_texts(raw_data_traloi_out)
data_traloi_in = traloi_tokenizer.texts_to_sequences(raw_data_traloi_in)
data_traloi_in = tf.keras.preprocessing.sequence.pad_sequences(data_traloi_in,
                                                           padding='post',maxlen=MAX_LEN_TRALOI)

data_traloi_out = traloi_tokenizer.texts_to_sequences(raw_data_traloi_out)
data_traloi_out = tf.keras.preprocessing.sequence.pad_sequences(data_traloi_out,
                                                            padding='post',maxlen=MAX_LEN_TRALOI)


num_samples = len(data_cauhoi)

SHUFFLE_BUFFER_SIZE = num_samples
BATCH_SIZE = 64
dataset = tf.data.Dataset.from_tensor_slices(
    (data_cauhoi, data_traloi_in, data_traloi_out))
dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)


def positional_embedding(pos, model_size):
    PE = np.zeros((1, model_size))
    for i in range(model_size):
        if i % 2 == 0:
            PE[:, i] = np.sin(pos / 10000 ** (i / model_size))
        else:
            PE[:, i] = np.cos(pos / 10000 ** ((i - 1) / model_size))
    return PE

max_length = max(len(data_cauhoi[0]), len(data_traloi_in[0]))
MODEL_SIZE = 128

pes = []
for i in range(max_length):
    pes.append(positional_embedding(i, MODEL_SIZE))

pes = np.concatenate(pes, axis=0)
pes = tf.constant(pes, dtype=tf.float32)


@register_keras_serializable()
class MultiHeadAttention(tf.keras.Model):
    def __init__(self, model_size, h):
        super(MultiHeadAttention, self).__init__()
        self.query_size = model_size // h
        self.key_size = model_size // h
        self.value_size = model_size // h
        self.h = h
        self.wq = [tf.keras.layers.Dense(self.query_size) for _ in range(h)]
        self.wk = [tf.keras.layers.Dense(self.key_size) for _ in range(h)]
        self.wv = [tf.keras.layers.Dense(self.value_size) for _ in range(h)]
        self.wo = tf.keras.layers.Dense(model_size)

    def call(self, decoder_output, encoder_output):
        # decoder_output has shape (batch, decoder_len, model_size)
        # encoder_output has shape (batch, encoder_len, model_size)
        heads = []
        for i in range(self.h):
            score = tf.matmul(self.wq[i](decoder_output), self.wk[i](encoder_output), transpose_b=True) / tf.math.sqrt(tf.dtypes.cast(self.key_size, tf.float32))
            # score has shape (batch, decoder_len, encoder_len)
            alignment = tf.nn.softmax(score, axis=2)
            # alignment has shape (batch, decoder_len, encoder_len)
            head = tf.matmul(alignment, self.wv[i](encoder_output))
            # head has shape (batch, decoder_len, value_size)
            heads.append(head)
        heads = tf.concat(heads, axis=2)
        heads = self.wo(heads)
        # heads has shape (batch, decoder_len, model_size)
        return heads
    
@register_keras_serializable()
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, model_size, num_layers, h):
        super(Encoder, self).__init__()
        self.model_size = model_size
        self.num_layers = num_layers
        self.h = h
        self.embedding = tf.keras.layers.Embedding(vocab_size, model_size)
        self.attention = [MultiHeadAttention(model_size, h) for _ in range(num_layers)]

        self.attention_norm = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]

        self.dense_1 = [tf.keras.layers.Dense(512, activation='relu') for _ in range(num_layers)]
        self.dense_2 = [tf.keras.layers.Dense(model_size) for _ in range(num_layers)]
        self.ffn_norm = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]

    def call(self, sequence):
        sub_in = []
        for i in range(sequence.shape[1]):
            embed = self.embedding(tf.expand_dims(sequence[:, i], axis=1))
            sub_in.append(embed + pes[i, :])

        sub_in = tf.concat(sub_in, axis=1)

        for i in range(self.num_layers):
            sub_out = []
            for j in range(sub_in.shape[1]):
                attention = self.attention[i](
                    tf.expand_dims(sub_in[:, j, :], axis=1), sub_in)

                sub_out.append(attention)

            sub_out = tf.concat(sub_out, axis=1)
            sub_out = sub_in + sub_out
            sub_out = self.attention_norm[i](sub_out)

            ffn_in = sub_out

            ffn_out = self.dense_2[i](self.dense_1[i](ffn_in))
            ffn_out = ffn_in + ffn_out
            ffn_out = self.ffn_norm[i](ffn_out)

            sub_in = ffn_out

        return ffn_out
@register_keras_serializable()
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, model_size, num_layers, h):
        super(Decoder, self).__init__()
        self.model_size = model_size
        self.num_layers = num_layers
        self.h = h
        self.embedding = tf.keras.layers.Embedding(vocab_size, model_size)
        self.attention_bot = [MultiHeadAttention(model_size, h) for _ in range(num_layers)]
        self.attention_bot_norm = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]
        self.attention_mid = [MultiHeadAttention(model_size, h) for _ in range(num_layers)]
        self.attention_mid_norm = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]

        self.dense_1 = [tf.keras.layers.Dense(512, activation='relu') for _ in range(num_layers)]
        self.dense_2 = [tf.keras.layers.Dense(model_size) for _ in range(num_layers)]
        self.ffn_norm = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]

        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, sequence, encoder_output):
        # EMBEDDING AND POSITIONAL EMBEDDING
        embed_out = self.embedding(sequence) # shape: (batch, sequence_len, model_size)
        embed_out = embed_out + pes[:tf.shape(sequence)[1], :] # Add positional encoding


        bot_sub_in = embed_out

        for i in range(self.num_layers):
            # BOTTOM MULTIHEAD SUB LAYER
            bot_sub_out = []

            for j in range(bot_sub_in.shape[1]):
                values = bot_sub_in[:, :j, :]
                attention = self.attention_bot[i](
                    tf.expand_dims(bot_sub_in[:, j, :], axis=1), values)

                bot_sub_out.append(attention)
            bot_sub_out = tf.concat(bot_sub_out, axis=1)
            bot_sub_out = bot_sub_in + bot_sub_out
            bot_sub_out = self.attention_bot_norm[i](bot_sub_out)

            # MIDDLE MULTIHEAD SUB LAYER
            mid_sub_in = bot_sub_out

            mid_sub_out = []
            for j in range(mid_sub_in.shape[1]):
                attention = self.attention_mid[i](
                    tf.expand_dims(mid_sub_in[:, j, :], axis=1), encoder_output)

                mid_sub_out.append(attention)

            mid_sub_out = tf.concat(mid_sub_out, axis=1)
            mid_sub_out = mid_sub_out + mid_sub_in
            mid_sub_out = self.attention_mid_norm[i](mid_sub_out)

            # FFN
            ffn_in = mid_sub_out

            ffn_out = self.dense_2[i](self.dense_1[i](ffn_in))
            ffn_out = ffn_out + ffn_in
            ffn_out = self.ffn_norm[i](ffn_out)

            bot_sub_in = ffn_out

        logits = self.dense(ffn_out)

        return logits


@register_keras_serializable()
class Seq2SeqModel(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, source_seq, target_seq_in):
        encoder_output = self.encoder(source_seq)
        decoder_output = self.decoder(target_seq_in, encoder_output)
        return decoder_output



