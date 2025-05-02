import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Descargar recursos necesarios de NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Configuración general
st.set_page_config(page_title="Análisis de Tuits", layout="wide")
st.markdown("<h1 style='text-align: center;'>🔍 Análisis de Sentimiento y Predicción de Viralidad</h1>", unsafe_allow_html=True)

# Cargar y limpiar el dataset
@st.cache_data
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
    texto = re.sub(r'@\w+|#', '', texto)
    texto = re.sub(r'[^a-z\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

# Tuits de entrenamiento manual (Positivos, Negativos, Neutrales)
tuits_entrenamiento = [
    # POSITIVOS
    ("Me encanta este producto, es excelente!", "Positive"),
    ("Estoy muy feliz con el servicio", "Positive"),
    ("Qué gran experiencia, los felicito", "Positive"),
    ("Un buen trabajo realizado", "Positive"),
    ("¡Qué maravilla!", "Positive"),
    ("Estoy satisfecho", "Positive"),
    ("¡Felicidades al equipo por el logro!", "Positive"),
    ("Todo fue perfecto, gracias!", "Positive"),
    ("Increíble atención al cliente", "Positive"),
    ("¡Excelente calidad, súper recomendado!", "Positive"),
    ("Muy útil y práctico", "Positive"),
    ("Me hicieron el día, gracias!", "Positive"),
    ("Servicio rápido y amable", "Positive"),
    ("Gran experiencia, volveré pronto", "Positive"),
    ("Estoy encantado con el resultado", "Positive"),

    # NEGATIVOS
    ("Esto es horrible, no me gustó nada", "Negative"),
    ("Una pérdida de tiempo total", "Negative"),
    ("Muy mal servicio, estoy decepcionado", "Negative"),
    ("Terrible atención al cliente", "Negative"),
    ("Lo odio", "Negative"),
    ("Pésima experiencia, no vuelvo más", "Negative"),
    ("Me arrepiento de haberlo comprado", "Negative"),
    ("Esto es inaceptable", "Negative"),
    ("Nada funciona como debería", "Negative"),
    ("No lo recomiendo para nada", "Negative"),
    ("Estoy muy molesto con el resultado", "Negative"),
    ("Una total decepción", "Negative"),
    ("Demasiado caro para lo que es", "Negative"),
    ("El peor servicio que he recibido", "Negative"),
    ("Nunca más compro aquí", "Negative"),

    # NEUTROS
    ("Es un producto", "Neutral"),
    ("No tengo una opinión clara", "Neutral"),
    ("Parece estar bien, supongo", "Neutral"),
    ("Ni bien ni mal", "Neutral"),
    ("Recibido el pedido", "Neutral"),
    ("Aún no lo he probado", "Neutral"),
    ("Información recibida correctamente", "Neutral"),
    ("Nada que destacar", "Neutral"),
    ("Mensaje enviado", "Neutral"),
    ("Estuve viendo el producto", "Neutral"),
    ("Solo quería preguntar una duda", "Neutral"),
    ("Actualización completada", "Neutral"),
    ("He leído el mensaje", "Neutral"),
    ("Producto recibido a tiempo", "Neutral"),
    ("Es aceptable, nada fuera de lo normal", "Neutral"),
]

# Crear DataFrame manual para los tuits de entrenamiento
df_manual = pd.DataFrame(tuits_entrenamiento, columns=['tweet', 'sentiment'])
df_manual['clean_tweet'] = df_manual['tweet'].apply(limpiar_texto)

# Preparación de los datos para el entrenamiento
vectorizer = TfidfVectorizer(max_features=1000)
X_manual = vectorizer.fit_transform(df_manual['clean_tweet'])
y_manual = df_manual['sentiment']

# Entrenamiento del modelo con los tuits manuales
model_sentiment = LogisticRegression()
model_sentiment.fit(X_manual, y_manual)

# Cargar y preparar datos
df = cargar_dataset()
df['clean_tweet'] = df['tweet'].apply(limpiar_texto)
df['retweets'] = np.random.randint(0, 200, size=len(df))
df['viral'] = (df['retweets'] > 100).astype(int)

# Filtrar solo sentimientos válidos
df = df[df['sentiment'].isin(['Positive', 'Negative', 'Neutral'])]

# Modelos
X = vectorizer.transform(df['clean_tweet'])
y_viral = df['viral']

X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(X, y_viral, test_size=0.2, random_state=42)

model_viral = LogisticRegression()
model_viral.fit(X_train_v, y_train_v)

# Sidebar
st.sidebar.markdown("## Escribe tu tuit ✍️")
tuit_input = st.sidebar.text_area("Ingresa un tuit para analizar:", height=150)

# Predicción del usuario
if tuit_input:
    clean_input = limpiar_texto(tuit_input)
    vec_input = vectorizer.transform([clean_input])
    pred_viral = model_viral.predict(vec_input)[0]
    pred_sentiment = model_sentiment.predict(vec_input)[0]

    col1, col2 = st.columns(2)
    with col1:
        if pred_viral == 1:
            st.success("🚀 Este tuit es probablemente **VIRAL**!")
        else:
            st.info("Este tuit **no parece viral**.")

    with col2:
        st.write(f"### 😄 Sentimiento detectado: **{pred_sentiment.capitalize()}**")
        if pred_sentiment == "Positive":
            st.success("Buen trabajo, suena positivo!")
        elif pred_sentiment == "Negative":
            st.error("Este tuit suena negativo 😟")
        else:
            st.warning("Es un tuit neutral.")

# Estadísticas generales
st.markdown("---")
st.markdown("## 📊 Estadísticas Generales")

col3, col4 = st.columns(2)

with col3:
    st.markdown("### Distribución de Sentimientos")
    fig_s, ax_s = plt.subplots()
    df['sentiment'].value_counts().plot(kind='bar', ax=ax_s, color=['green', 'red', 'blue'])
    ax_s.set_title("Sentimientos")
    st.pyplot(fig_s)

with col4:
    st.markdown("### Distribución de Tuits Virales")
    fig_v, ax_v = plt.subplots()
    df['viral'].value_counts().plot(kind='bar', ax=ax_v, color=['orange', 'blue'])
    ax_v.set_title("Viral vs No Viral")
    st.pyplot(fig_v)

# WordCloud
st.markdown("### ☁️ Nube de Palabras de Tuits")
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['clean_tweet']))
fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
ax_wc.imshow(wordcloud, interpolation='bilinear')
ax_wc.axis('off')
st.pyplot(fig_wc)

# Precisión del modelo
st.markdown("### 🧠 Precisión de los Modelos")
st.write(f"**Precisión de Viralidad**: {accuracy_score(y_test_v, model_viral.predict(X_test_v)):.2f}")
st.write(f"**Precisión de Sentimiento**: {accuracy_score(y_manual, model_sentiment.predict(X_manual)):.2f}")






