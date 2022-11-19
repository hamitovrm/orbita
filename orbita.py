import io
import requests
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions

API_URL_ta = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-mul"
headers = {"Authorization": f"Bearer {'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'}"}

def translate(payload, API_URL):
	response = requests.post(API_URL, headers=headers, json=payload )
	return response.json

@st.cache(allow_output_mutation=True)
def load_model():
    return EfficientNetB7(weights='imagenet')

def preprocess_image(img):
    img = img.resize((600, 600))
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
        en_text=str(cl[1]).replace('_',' ')
        en_text=''+en_text
        trans_ta = translate({"inputs": [">>rus<< "+en_text, ">>tat<< "+en_text, ">>deu<< "+en_text,],}, API_URL_ta)
        tr_test=tuple(trans_ta())
        outstr= str(int(cl[2]*100))+ '% это \t eng: '+ str(en_text),'\t rus: '+ str(tr_test[0]['translation_text'])+'\t tat: '+ str(tr_test[1]['translation_text'])+'\t deu: '+ str(tr_test[2]['translation_text'])
        st.write(outstr)
     

model = load_model()

st.title('Распознавание объектов с переводом на разные языки')
img = load_image()
result = st.button('Распознать изображение')
if result:
    x = preprocess_image(img)
    preds = model.predict(x)
    st.write('С вероятностью:')
    print_predictions(preds)

