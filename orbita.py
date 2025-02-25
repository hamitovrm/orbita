import io
import requests
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions

API_URL_ta = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-mul"
headers = {"Authorization": f"Bearer {'HugginFace_token'}"}

def translate(payload, API_URL):
	response = requests.post(API_URL, headers=headers, json=payload )
	return response.json

@st.cache_resource
def load_model():
    return EfficientNetB7(weights='imagenet')

def preprocess_image(img):
    imgr = img.resize((600, 600))
    x = image.img_to_array(imgr)
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
        en_text=str(cl[1]).replace('_',' ')
        en_text=''+en_text
        trans_ta = translate({"inputs": [">>rus<< "+en_text, ">>tat<< "+en_text,],}, API_URL_ta)
        tr_test=tuple(trans_ta())
        col1, col2, col3, col4 = st.columns(4)

        with col1:
          st.subheader(" ")
          st.write(str(int(cl[2]*100))+"%")

        with col2:
          st.subheader("eng")
          st.write(str(en_text))

        with col3:
          st.subheader("rus")
          st.write(str(tr_test[0]["translation_text"]))

        with col4:
          st.subheader("tat")
          st.write(str(tr_test[1]["translation_text"]))


model = load_model()

st.title('ORBiTA')
st.write('Object Reconginion Bi Translation Application')
img = load_image()
result = st.button('Распознать изображение')
if result:
    x = preprocess_image(img)
    preds = model.predict(x)
    st.write('С вероятностью:')
    print_predictions(preds)

