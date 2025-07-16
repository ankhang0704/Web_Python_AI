import os
import pickle
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import json
import time
from django.conf import settings
# ...existing code...
# ========================================================== 
# PHẦN TẢI MÔ HÌNH THỰC TẾ
# ==========================================================

MODEL_LOADED = False
encoder_model = None
decoder_model = None
tokenizer_input = None  # Tokenizer cho câu hỏi
tokenizer_output = None  # Tokenizer cho câu trả lời
max_input_length = 100
max_output_length = 100

def load_ai_model():
    """
    Tải mô hình AI thực tế từ thư mục đã chỉ định
    """
    global MODEL_LOADED, encoder_model, decoder_model, tokenizer_input, tokenizer_output
    global max_input_length, max_output_length
    
    if MODEL_LOADED:
        print("--- Mô hình đã được tải trước đó ---")
        return
    
    try:
        print("--- Đang tải mô hình AI thực tế... ---")
        
        # Đường dẫn tới thư mục chứa mô hình
        model_base_path = os.path.join(settings.BASE_DIR, 'ai_model_chats', 'my_seq2seq_chatbot')
        
        # Kiểm tra xem thư mục có tồn tại không
        if not os.path.exists(model_base_path):
            print(f"⚠️ Không tìm thấy thư mục mô hình: {model_base_path}")
            print("Sẽ tạo thư mục và sử dụng mô hình mặc định")
            os.makedirs(model_base_path, exist_ok=True)
            create_fallback_model()
            return
        # Tải Tokenizer cho câu hỏi
        tokenizer_input_path = os.path.join(model_base_path, "cauhoi_tokenizer.pkl")
        if os.path.exists(tokenizer_input_path):
            print("Đang tải Tokenizer câu hỏi...")
            # Thử tải dưới dạng pickle hoặc json
            try:
                with open(tokenizer_input_path, 'rb') as f:
                    tokenizer_input = pickle.load(f)
                print("✓ Tokenizer câu hỏi đã được tải (pickle)")
            except:
                try:
                    with open(tokenizer_input_path, 'r', encoding='utf-8') as f:
                        tokenizer_data = json.load(f)
                    tokenizer_input = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_data)
                    print("✓ Tokenizer câu hỏi đã được tải (json)")
                except:
                    print("⚠️ Không thể tải Tokenizer câu hỏi, sẽ tạo mặc định")
                    tokenizer_input = create_default_tokenizer()
        else:
            print("⚠️ Không tìm thấy Tokenizer câu hỏi, sẽ tạo mặc định")
            tokenizer_input = create_default_tokenizer()
        
        # Tải Tokenizer cho câu trả lời
        tokenizer_output_path = os.path.join(model_base_path, "traloi_tokenizer.pkl")
        if os.path.exists(tokenizer_output_path):
            print("Đang tải Tokenizer trả lời...")
            try:
                with open(tokenizer_output_path, 'rb') as f:
                    tokenizer_output = pickle.load(f)
                print("✓ Tokenizer trả lời đã được tải (pickle)")
            except:
                try:
                    with open(tokenizer_output_path, 'r', encoding='utf-8') as f:
                        tokenizer_data = json.load(f)
                    tokenizer_output = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_data)
                    print("✓ Tokenizer trả lời đã được tải (json)")
                except:
                    print("⚠️ Không thể tải Tokenizer trả lời, sẽ tạo mặc định")
                    tokenizer_output = create_default_tokenizer()
        else:
            print("⚠️ Không tìm thấy Tokenizer trả lời, sẽ tạo mặc định")
            tokenizer_output = create_default_tokenizer()
            
        # Tải Encoder
        encoder_path = os.path.join(model_base_path, "encoder_weights.weights.h5")
        if os.path.exists(encoder_path):
            print("Đang tải Encoder...")
            encoder_model = create_default_encoder()
            dummy_input = tf.zeros((1, max_input_length), dtype=tf.int32)
            encoder_model(dummy_input)  # hoặc shape phù hợp với dữ liệu của bạn
            try:
                encoder_model.load_weights(encoder_path)
                print("✓ Encoder đã được tải thành công")
            except Exception as e:
                print(f"⚠️ Lỗi khi tải trọng số Encoder: {e}")
                print("Sẽ sử dụng encoder mặc định")
                encoder_model = create_default_encoder()
        else:
            print("⚠️ Không tìm thấy Encoder, sẽ tạo mô hình mặc định")
            encoder_model = create_default_encoder()
        
        # Tải Decoder
        decoder_path = os.path.join(model_base_path, "decoder_weights.weights.h5")
        if os.path.exists(decoder_path):
            print("Đang tải Decoder...")
            decoder_model = create_default_decoder()
            dummy_seq = tf.zeros((1, max_output_length), dtype=tf.int32)
            dummy_enc = tf.zeros((1, max_input_length, MODEL_SIZE), dtype=tf.float32)
            decoder_model(dummy_seq, dummy_enc)
            try:
                decoder_model.load_weights(decoder_path)
                print("✓ Decoder đã được tải thành công")
            except Exception as e:
                print(f"⚠️ Lỗi khi tải trọng số Decoder: {e}")
                print("Sẽ sử dụng decoder mặc định")
                decoder_model = create_default_decoder()
        else:
            print("⚠️ Không tìm thấy Decoder, sẽ tạo mô hình mặc định")
            decoder_model = create_default_decoder()
        
        
        
        # # Tải cấu hình độ dài tối đa (nếu có)
        # config_path = os.path.join(model_path, "config.json")
        # if os.path.exists(config_path):
        #     with open(config_path, 'r', encoding='utf-8') as f:
        #         config = json.load(f)
        #         max_input_length = config.get('max_input_length', 100)
        #         max_output_length = config.get('max_output_length', 100)
        #         print(f"✓ Cấu hình tải thành công: max_input={max_input_length}, max_output={max_output_length}")
        
        MODEL_LOADED = True
        print("--- Mô hình AI đã sẵn sàng! ---")
        
    except Exception as e:
        print(f"--- LỖI KHI TẢI MÔ HÌNH: {e} ---")
        print("--- Sẽ sử dụng mô hình mặc định ---")
        create_fallback_model()
        MODEL_LOADED = True

# Thông số đồng bộ với file huấn luyện
MODEL_SIZE = 256
NUM_LAYERS = 2
H = 2

def positional_embedding(pos, model_size):
    PE = np.zeros((1, model_size))
    for i in range(model_size):
        if i % 2 == 0:
            PE[:, i] = np.sin(pos / 10000 ** (i / model_size))
        else:
            PE[:, i] = np.cos(pos / 10000 ** ((i - 1) / model_size))
    return PE

max_length = max(max_input_length, max_output_length)
pes = []
for i in range(max_length):
    pes.append(positional_embedding(i, MODEL_SIZE))
pes = np.concatenate(pes, axis=0)
pes = tf.constant(pes, dtype=tf.float32)


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

# Positional encoding dummy (replace with your actual positional encoding)

def create_default_encoder():
    vocab_size = len(tokenizer_input.word_index) + 1
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
    return Encoder(vocab_size, MODEL_SIZE, NUM_LAYERS, H)

def create_default_decoder():
    vocab_size = len(tokenizer_output.word_index) + 1
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
            embed_out = []
            for i in range(sequence.shape[1]):
                embed = self.embedding(tf.expand_dims(sequence[:, i], axis=1))
                embed_out.append(pes[i, :] + embed)
            embed_out = tf.concat(embed_out, axis=1)
            bot_sub_in = embed_out
            for i in range(self.num_layers):
                bot_sub_out = []
                for j in range(bot_sub_in.shape[1]):
                    values = bot_sub_in[:, :j, :]
                    attention = self.attention_bot[i](
                        tf.expand_dims(bot_sub_in[:, j, :], axis=1), values)
                    bot_sub_out.append(attention)
                bot_sub_out = tf.concat(bot_sub_out, axis=1)
                bot_sub_out = bot_sub_in + bot_sub_out
                bot_sub_out = self.attention_bot_norm[i](bot_sub_out)
                mid_sub_in = bot_sub_out
                mid_sub_out = []
                for j in range(mid_sub_in.shape[1]):
                    attention = self.attention_mid[i](
                        tf.expand_dims(mid_sub_in[:, j, :], axis=1), encoder_output)
                    mid_sub_out.append(attention)
                mid_sub_out = tf.concat(mid_sub_out, axis=1)
                mid_sub_out = mid_sub_out + mid_sub_in
                mid_sub_out = self.attention_mid_norm[i](mid_sub_out)
                ffn_in = mid_sub_out
                ffn_out = self.dense_2[i](self.dense_1[i](ffn_in))
                ffn_out = ffn_out + ffn_in
                ffn_out = self.ffn_norm[i](ffn_out)
                bot_sub_in = ffn_out
            logits = self.dense(ffn_out)
            return logits
    return Decoder(vocab_size, MODEL_SIZE, NUM_LAYERS, H)

def create_default_tokenizer():
    """Tạo tokenizer mặc định"""
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000, oov_token='<OOV>')
    # Thêm một số từ cơ bản
    tokenizer.fit_on_texts(['xin chào', 'tạm biệt', 'cảm ơn', 'điện thoại', 'giá cả'])
    return tokenizer

def create_fallback_model():
    """Tạo mô hình dự phòng đơn giản"""
    global encoder_model, decoder_model, tokenizer_input, tokenizer_output
    
    print("Tạo mô hình dự phòng...")
    encoder_model = create_default_encoder()
    decoder_model = create_default_decoder()
    tokenizer_input = create_default_tokenizer()
    tokenizer_output = create_default_tokenizer()

# ========================================================== 
# HÀM DỰ ĐOÁN SỬ DỤNG MÔ HÌNH THỰC TẾ
# ==========================================================

def preprocess_input(text):
    """Tiền xử lý văn bản đầu vào"""
    # Chuyển về chữ thường
    text = text.lower().strip()
    
    # Loại bỏ các ký tự đặc biệt (tùy chọn)
    import re
    text = re.sub(r'[^\w\s]', '', text)
    
    return text

def predict_with_model(user_input):
    try:
        processed_input = preprocess_input(user_input)
        input_seq = tokenizer_input.texts_to_sequences([processed_input])
        input_seq = pad_sequences(input_seq, maxlen=max_input_length, padding='post')

        # Encode đầu vào
        encoder_output = encoder_model(input_seq)  # đúng kiểu transformer

        # Khởi tạo target sequence
        target_seq = np.zeros((1, max_output_length), dtype=np.int32)
        target_seq[0, 0] = tokenizer_output.word_index.get('<start>', 1)

        decoded_sentence = ''
        for i in range(1, max_output_length):
            # Dự đoán tiếp theo
            output_tokens = decoder_model(target_seq, encoder_output)
            sampled_token_index = np.argmax(output_tokens[0, i-1, :])

            sampled_word = None
            for word, index in tokenizer_output.word_index.items():
                if index == sampled_token_index:
                    sampled_word = word
                    break

            if sampled_word is None or sampled_word == '<end>':
                break

            if sampled_word != '<start>':
                decoded_sentence += sampled_word + ' '

            target_seq[0, i] = sampled_token_index  # cập nhật token tiếp theo

        return decoded_sentence.strip()

    except Exception as e:
        print(f"Lỗi trong quá trình dự đoán: {e}")
        return None

def predict(user_input: str) -> str:
    """
    Hàm dự đoán chính sử dụng mô hình AI thực tế
    """
    print(f"Nhận được câu hỏi từ người dùng: '{user_input}'")
    
    # Kiểm tra mô hình đã được tải chưa
    if not MODEL_LOADED:
        load_ai_model()
    
    try:
        # Sử dụng mô hình thực tế để dự đoán
        ai_response = predict_with_model(user_input)
        
        # Nếu mô hình trả về kết quả hợp lệ
        if ai_response and len(ai_response.strip()) > 0:
            print(f"AI trả lời: '{ai_response}'")
            return ai_response
        else:
            # Fallback responses dựa trên từ khóa (như backup)
            return fallback_response(user_input)
            
    except Exception as e:
        print(f"--- LỖI XẢY RA TRONG HÀM PREDICT: {e} ---")
        return fallback_response(user_input)

def fallback_response(user_input):
    """Hàm trả lời dự phòng khi mô hình không hoạt động"""
    user_input_lower = user_input.lower()
    
    if any(keyword in user_input_lower for keyword in ['xin chào', 'chào', 'hello', 'hi']):
        return "Xin chào! Tôi có thể giúp gì cho bạn hôm nay?"
    
    elif any(keyword in user_input_lower for keyword in ['điện thoại', 'phone', 'smartphone']):
        return "Chúng tôi có nhiều loại điện thoại với giá cả phù hợp. Bạn quan tâm đến loại nào?"
    
    elif any(keyword in user_input_lower for keyword in ['giá', 'price', 'cost', 'tiền']):
        return "Về giá cả, chúng tôi có nhiều mức giá khác nhau. Bạn có thể cho biết sản phẩm cụ thể không?"
    
    elif any(keyword in user_input_lower for keyword in ['cảm ơn', 'thank', 'thanks']):
        return "Không có gì! Tôi luôn sẵn sàng giúp đỡ bạn."
    
    elif any(keyword in user_input_lower for keyword in ['tạm biệt', 'bye', 'goodbye']):
        return "Tạm biệt! Chúc bạn một ngày tốt lành!"
    
    else:
        return "Xin lỗi, tôi chưa hiểu rõ câu hỏi của bạn. Bạn có thể hỏi lại hoặc liên hệ với bộ phận hỗ trợ để được giúp đỡ tốt hơn."

# ========================================================== 
# KHỞI TẠO MÔ HÌNH KHI IMPORT MODULE
# ==========================================================

# Tự động tải mô hình khi module được import
print("Khởi tạo AI Model Handler...")
load_ai_model()