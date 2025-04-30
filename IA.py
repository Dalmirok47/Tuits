import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit as st

# Descargar recursos de NLTK si es necesario
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Cargar y limpiar el dataset (esto se hará cuando la app lo necesite)
@st.cache_data  # Usar cache_data en lugar de obsoleto st.cache
def cargar_dataset():
    df = pd.read_csv('twitter_training.csv')
    df.columns = ['id', 'entity', 'sentiment', 'tweet']
    df['tweet'] = df['tweet'].astype(str)
    df = df[df['tweet'].notnull()]
    return df

# Función de limpieza de texto
def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r"http\S+|www\S+|https\S+", '', texto)
    texto = re.sub(r'\@w+|\#','', texto)
    texto = re.sub(r'[^A-Za-z\s]', '', texto)
    return texto

# Cargar datos
df = cargar_dataset()

# Limpiar los tuits
df['clean_tweet'] = df['tweet'].apply(limpiar_texto)

# Simular retweets (esto es solo para entrenamiento)
df['retweets'] = np.random.randint(0, 200, size=len(df))
df['viral'] = (df['retweets'] > 100).astype(int)

# Vectorizar los tuits
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['clean_tweet'])
y = df['viral']

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluación
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Título y descripción
st.title("Predicción de Tuits Virales")
st.write("""
    Esta app predice si un tuit será viral o no, según la cantidad de retuits simulados y su análisis de sentimientos.
    Además, muestra estadísticas sobre tuits previos y cómo los sentimientos afectan la viralidad.
""")

# **Predicción de Tuit Individual**
tuit_input = st.text_area("Escribe un tuit para predecir si es viral:", height=150)

if tuit_input:
    clean_input = limpiar_texto(tuit_input)
    tuit_vec = vectorizer.transform([clean_input])
    prediction = model.predict(tuit_vec)
    if prediction == 1:
        st.success("Este tuit **es viral**! 🎉")
    else:
        st.warning("⚠️ Este tuit **no es viral**.")

# **Estadísticas de Tuits**
st.subheader("Estadísticas de Tuits Pasados")

# Mostrar cantidad de tuits por sentimiento
sentimiento_count = df['sentiment'].value_counts()
st.write(f"**Cantidad de tuits por sentimiento**: {sentimiento_count.to_dict()}")

# Mostrar cantidad de tuits virales y no virales
viral_count = df['viral'].value_counts()
st.write(f"**Cantidad de tuits virales y no virales**: {viral_count.to_dict()}")

# **Distribución de Sentimientos**
st.subheader("Distribución de Sentimientos")
fig, ax = plt.subplots(figsize=(8, 6))
sentimiento_count.plot(kind='bar', ax=ax, color=['green', 'red', 'blue'])
ax.set_xlabel("Sentimiento")
ax.set_ylabel("Cantidad de Tuits")
ax.set_title("Distribución de Sentimientos de los Tuits")
st.pyplot(fig)

# **Distribución de Viralidad**
st.subheader("Distribución de Viralidad")
viral_count = df['viral'].value_counts()
fig_viral, ax_viral = plt.subplots(figsize=(8, 6))
viral_count.plot(kind='bar', ax=ax_viral, color=['orange', 'blue'])
ax_viral.set_xlabel("Viralidad (1: Viral, 0: No Viral)")
ax_viral.set_ylabel("Cantidad de Tuits")
ax_viral.set_title("Distribución de Viralidad")
st.pyplot(fig_viral)

# **WordCloud de los Tuits**
st.subheader("Palabras Más Frecuentes en los Tuits")
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['clean_tweet']))
fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
ax_wc.imshow(wordcloud, interpolation='bilinear')
ax_wc.axis('off')
ax_wc.set_title("WordCloud de Tuits", fontsize=16)
st.pyplot(fig_wc)

# **Mostrar Precisión del Modelo**
st.write(f"**Precisión del modelo**: {accuracy:.2f}")



