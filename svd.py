import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from skimage import io
import streamlit as st

st.header('SVD для изображений', divider=True)
st.write("""
    #### 1. Загрузи свою картинку
         """)

image_url = st.text_input("Введите URL изображения:")

if image_url:
    st.image(image_url, caption='Ваше изображение', use_column_width=True)
else:
    image_url = 'https://zvetnoe.ru/upload/resize_cache/catalog/w750h1000/2db/2db1856cf3254e29aa2e4925d6312b80.jpg'
    st.image(image_url, caption='Ваше изображение', use_column_width=True)


image = io.imread(image_url)[:, :, 2]
U, sing_vals, V = np.linalg.svd(image)

sigma = np.zeros(shape = image.shape)
np.fill_diagonal(sigma, sing_vals)
st.write("""
    #### 2. Разложили на простые матрицы 
         """)
U.shape, sigma.shape, V.shape

max_k = max(list(image.shape))
top_k = st.sidebar.slider(
    "Выберите степень шакалистости:",
    min_value=0,      # Минимальное значение
    max_value=max_k,    # Максимальное значение
    value=50,         # Значение по умолчанию
    step=1            # Шаг изменения
)

st.write(f'Cохранил {100 * top_k / len(sing_vals)}% данных')

st.write("""
    #### 3. Выши шакалы готовы
         """)

trunc_U = U[:, :top_k]
trunc_sigma = sigma[:top_k, :top_k]
trunc_V = V[:top_k, :]
fig, axes = plt.subplots( figsize=(20,10))

axes.imshow(trunc_U@trunc_sigma@trunc_V, cmap='grey')
axes.set_title(f'top_k = {top_k} компонент')
st.pyplot(fig)

