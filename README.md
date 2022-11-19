#  Object Reconginion Bi Translation Application

[ORBiTA](https://orbita.streamlit.app/) - Web-приложение для распознавания объектов на изображении и параллельного перевода на два языка: русский, татарский. 

Используются библиотеки:

- [TensorFlow](https://www.tensorflow.org/).
- [Streamlit](https://streamlit.io/).
- [Requests](https://requests.readthedocs.io/en/latest/)
- [NumPy](https://numpy.org/)

Для распознавания изображений используется нейронная сеть [EfficientNet7](https://keras.io/api/applications/efficientnet/#efficientnetb7-function). Подробности о модели в статье:

- [Новые архитектуры нейросетей](https://habr.com/ru/post/498168/#EfficientNet).

Для перевода распознаных объектов на разные языки используется модель [Language Technology Research Group at the University of Helsinki](https://huggingface.co/Helsinki-NLP)


