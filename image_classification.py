import io
import requests
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions


API_URL_ru = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-ru"
API_URL_fr = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-fr"
headers = {"Authorization": f"Bearer {'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'}"}

def translate(payload, API_URL):
	response = requests.post(API_URL, headers=headers, json=payload )
	return response.json
	

@st.cache(allow_output_mutation=True)
def load_model():
    return EfficientNetB0(weights='imagenet')


def preprocess_image(img):
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def load_image():
    uploaded_file = st.file_uploader(label='Выберите изображение для распознавания')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None


def print_predictions(preds):
    classes = decode_predictions(preds, top=3)[0]
    for cl in classes:
        st.write(str(cl[1]).replace('_'," "), cl[2])
        trans_ru = translate({"inputs": str(cl[1]).replace('_'," "),}, API_URL_ru)
        for tt in trans_ru():
             st.write(str(tt['translation_text']))
        trans_fr = translate({"inputs": str(cl[1]).replace('_'," "),}, API_URL_fr)
        for tt in trans_fr():
             st.write(str(tt['translation_text']))


model = load_model()

st.title('Классификация изображений с переводом на разные языки')
img = load_image()
result = st.button('Распознать изображение')
if result:
    x = preprocess_image(img)
    preds = model.predict(x)
    st.write('**Результаты распознавания:**')
    print_predictions(preds)

