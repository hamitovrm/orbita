import io
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
from transformers import MarianMTModel, MarianTokenizer
from transformers import pipeline

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
        st.write(cl[1], cl[2])
        


def print_translation(preds):
    classes = decode_predictions(preds, top=3)[0]
    for cl in classes:
        tr=translator(str(cl[1]))
        st.write(tr)
   # src_text = [
   # ">>tat<< this is a sentence in english that we want to translate to tatar",
   # ">>tat<< Sit down and eat soup.",
   # ]   
   # translated = model_tr.generate(**tokenizer(src_text, return_tensors="pt", padding=True))
   # [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    


model = load_model()

#model_name = "Helsinki-NLP/opus-mt-en-mul"
#tokenizer = MarianTokenizer.from_pretrained(model_name)
#model_tr = MarianMTModel.from_pretrained(model_name)

translator = pipeline("translation_en_to_ru", "Helsinki-NLP/opus-mt-en-ru")


st.title('Классификация изображений с переводом на русский и татарский языки')
img = load_image()
result = st.button('Распознать изображение')
if result:
    x = preprocess_image(img)
    preds = model.predict(x)
    st.write('**Результаты распознавания:**')
    print_predictions(preds)
    print_translation(preds)
