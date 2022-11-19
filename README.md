#  Object Reconginion Bi Translation Application

[ORBiTA](https://orbita.streamlit.app/) - Web-приложение для распознавания объектов на изображении и параллельного перевода на два языка: русский, татарский. 

Используются библиотеки:

- [TensorFlow](https://www.tensorflow.org/).
- [Streamlit](https://streamlit.io/).
- [Requests](https://requests.readthedocs.io/en/latest/)
- [NumPy](https://numpy.org/)
- [PIL](http://www.pythonware.com/products/pil/)

Для распознавания изображений используется нейронная сеть [EfficientNet7](https://keras.io/api/applications/efficientnet/#efficientnetb7-function). Подробности о модели в статьях:

- [Новые архитектуры нейросетей](https://habr.com/ru/post/498168/#EfficientNet).
[Что такое модель EfficientNet](https://russianblogs.com/article/97411642918/)

Для перевода распознаных объектов на разные языки используется модель Language Technology Research Group at the University of Helsinki [Helsinki-NLP/opus-mt-en-mul](https://huggingface.co/Helsinki-NLP/opus-mt-en-mul?text=My+name+is+Sarah+and+I+live+in+London)

Приложение развернуто по адресу https://orbita.streamlit.app/

