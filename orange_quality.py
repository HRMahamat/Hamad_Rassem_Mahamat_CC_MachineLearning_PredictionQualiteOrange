import joblib
import base64
import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest


result = {
    0: "faible qualité ; 🔴 voir mauvaise ",
    1: "qualité standard ; 🟡 tout ce qui a de plus basique",
    2: "bonne qualité ; 🟢 voir excellente"
}
pictures = ["faible", "moyenne", "elevee"]
colors = ["#FF4B4B", "#FFA500", "#00C853"]
model = joblib.load("Hamad_Rassem_Mahamat_The_final_model.pkl")
with open("Travis Scott - SICKO MODE ft. Drake (320 kbps).mp3", "rb") as f:
    audio_bytes = f.read()
b64 = base64.b64encode(audio_bytes).decode()


def input():
    size = st.sidebar.slider('Taille (cm)', 3.8, 12.0, 6.0)
    weight = st.sidebar.slider('Poids (g)', 50, 350, 175)
    brix = st.sidebar.slider('°Brix (saccharosité)', 6.0, 20.0, 11.8)
    ph = st.sidebar.slider('pH (acidité)', 2.9, 4.6, 4.0)
    softness = st.sidebar.slider('Echelle de tendreté', 1.0, 5.0, 3.0)
    day = st.sidebar.slider('Délai de récolte (jours)', 180, 420, 365)
    ripeness = st.sidebar.slider('Echelle de maturité', 1.0, 5.0, 2.0)
    deeporange = st.sidebar.slider('Couleur orange foncé', 0, 1, 1)
    blemishfree = st.sidebar.slider('Absence de défaut (0 = avec défaut et 1 = sans défaut)', 0, 1, 0)
    data={
        'Size (cm)': size,
        'Weight (g)': weight,
        'Brix (Sweetness)': brix,
        'pH (Acidity)': ph,
        'Softness (1-5)': softness,
        'HarvestTime (days)': day,
        'Ripeness (1-5)': ripeness,
        'DeepOrange (Color)': deeporange,
        'BlemishFree': blemishfree
    }
    parametres=pd.DataFrame(data, index=[0])
    return parametres


st.set_page_config(page_title="Qualité d'Orange 🍊", page_icon="🍊", layout="wide")

st.markdown("""
    <style>
        .stApp {
            background-color: #121212;
            color: #FFFFFF;
            font-family: 'Segoe UI', sans-serif;
        }
        h1, h2, h3 {
            color: #FF8C00;
        }
        .stSidebar {
            background-color: #1e1e1e;
            border-right: 2px solid #FF8C00;
        }
        .block-container {
            padding: 2rem;
        }
        .stDataFrame table {
            background-color: #1E1E1E;
            color: #FFFFFF;
        }
        .result-box {
            padding: 1.5rem;
            background-color: #1e1e1e;
            border-left: 8px solid #FF8C00;
            border-radius: 10px;
            margin-top: 1.5rem;
        }
        .result-box span {
            font-size: 1.3rem;
            font-weight: bold;
        }
        html, body, .main, .block-container {
            background-color: #121212 !important;
            color: #FFFFFF !important;
        }
        [data-testid="stSidebar"] > div:first-child {
            background-color: #1E1E1E !important;
        }
        h1, h2, h3 {
            color: #FF8C00 !important;
        }
        div[data-testid="stDataFrame"] div[role="grid"] {
            background-color: #1E1E1E !important;
            color: #FFFFFF !important;
        }
        div[data-testid="stAlert"] {
            background-color: #2E2E2E !important;
            color: #FFFFFF !important;
        }
    </style>
    """, unsafe_allow_html=True)

st.title("Bienvenue sur votre site de Machine Learning")
st.markdown("Nous prédisons la qualité du fruit **Orange** grâce à un modèle intelligent.")

st.sidebar.header("⚙️ Les paramètres d'entrées")
df=input()
st.sidebar.audio(audio_bytes, format="audio/mp3", start_time=0, autoplay=True, loop=True)

st.markdown("### <br>Vous souhaitez déterminer la <span style='color: #FFA500;'>qualité</span> de votre orange 🔍", unsafe_allow_html=True)
st.write(df.iloc[:, :6])
st.write(df.iloc[:, -3:])

predict = model.predict(df)[0]
st.markdown(f"""<br><div class="result-box"><span style='color: {colors[predict]};'>✅ Prédiction achevée!</span></div>""", unsafe_allow_html=True)
st.markdown("Nous estimons que l'orange que vous venez de décrire est de <span style='color: "+colors[predict]+";'>"+(str(result.get(predict, str(predict)))).upper()+"</span> ("+str(predict)+")!", unsafe_allow_html=True)
img=Image.open(pictures[predict]+".png")
st.image(img.resize((1000, int((float(img.size[1]) * float((370 / float(img.size[0])))))), Image.FILTERED), caption="Image de reférence", use_container_width=False)
