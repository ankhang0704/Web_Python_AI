# import re
# import os
# from django.utils.timezone import datetime
# from django.http import HttpResponse
# from django.shortcuts import render
# from keras.models import load_model
# import numpy as np
import os
from django.shortcuts import render
from django.conf import settings
from keras.models import load_model, Model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.preprocessing.sequence import pad_sequences
from django.http import HttpResponse
from django.utils.timezone import datetime
from pickle import load
import numpy as np
from PIL import Image


# Load model 1 lần
model_path = os.path.join(settings.BASE_DIR, 'model', 'model_v1.keras')
model = load_model(model_path)

cnn_model = InceptionV3(weights='imagenet')
cnn_model_new = Model(cnn_model.input, cnn_model.layers[-2].output)

wordtoix = load(open(os.path.join(settings.BASE_DIR, 'model', 'wordtoix.pkl'), 'rb'))
ixtoword = load(open(os.path.join(settings.BASE_DIR, 'model', 'ixtoword.pkl'), 'rb'))

max_length = 35   # đặt đúng với max_length bạn đã dùng

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def encode_image(img_path):
    img = preprocess_image(img_path)
    fea_vec = cnn_model_new.predict(img)
    fea_vec = np.reshape(fea_vec, (1, 2048))
    return fea_vec

def generate_caption(photo):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword.get(yhat)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()[1:-1]
    return ' '.join(final)

def index(request):
    caption = None
    image_url = None

    if request.method == 'POST' and 'image' in request.FILES:
        uploaded_file = request.FILES['image']
        save_path = os.path.join(settings.MEDIA_ROOT, uploaded_file.name)

        with open(save_path, 'wb+') as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)

        photo = encode_image(save_path)
        caption = generate_caption(photo)
        image_url = settings.MEDIA_URL + uploaded_file.name

    return render(request, 'Web_AI/index.html', {
        'caption': caption,
        'image_url': image_url
    })



def home(request):
    return render(request, "Web_AI/home.html")

def hello_there(request, name):
    print(request.build_absolute_uri()) #optional
    return render(
        request,
        'Web_AI/hello_there.html',
        {
            'name': name,
            'date': datetime.now()
        }
    )

def about(request):
    return render(request, "Web_AI/about.html")

def contact(request):
    return render(request, "Web_AI/contact.html")
