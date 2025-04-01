# app.py — Interface Streamlit complète : Illustrer son émotion – image / musique

import streamlit as st
import uuid
import os
import subprocess
from pathlib import Path
from transformers import pipeline
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Configuration Streamlit
st.set_page_config(
    page_title="Illustrer son émotion – image / musique",
    page_icon="🎼",
    layout="centered"
)

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Chargement des modèles
@st.cache_resource
def load_models():
    classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)
    sd = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    sd.to("cuda" if torch.cuda.is_available() else "cpu")
    return classifier, sd

emotion_classifier, sd_pipe = load_models()

# En-tête
st.markdown(
    """
    <style>
    .title { text-align: center; font-size: 2.8rem; font-weight: 700; color: #2c3e50; }
    .subtitle { text-align: center; font-size: 1.2rem; color: #7f8c8d; margin-bottom: 2rem; }
    </style>
    <div class="title">Illustrer son émotion</div>
    <div class="subtitle">Générez une image et une musique à partir de votre récit émotionnel</div>
    """,
    unsafe_allow_html=True
)

st.write("---")

# Formulaire utilisateur
with st.form("emotion_form"):
    titre = st.text_input("Titre de votre émotion (facultatif)", "Mon ressenti du jour")
    texte = st.text_area("Exprimez-vous librement", "Décrivez ici une situation ou un ressenti fort…")
    intensite = st.slider("Intensité émotionnelle perçue", 0.1, 2.0, 1.0, step=0.1)
    style = st.selectbox("Style visuel préféré", ["abstrait", "impressionniste", "photographique", "surréaliste", "aquarelle"])
    generer = st.form_submit_button("Illustrer mon émotion")

if generer and texte.strip():
    with st.spinner("Analyse et création en cours..."):
        detection = emotion_classifier(texte)[0]
        emotion = detection['label'].lower()

        id_unique = str(uuid.uuid4())
        midi_path = OUTPUT_DIR / f"{id_unique}.mid"
        wav_path = OUTPUT_DIR / f"{id_unique}.wav"
        mp3_path = OUTPUT_DIR / f"{id_unique}.mp3"
        partition_path = OUTPUT_DIR / f"{id_unique}.png"
        image_path = OUTPUT_DIR / f"{id_unique}_image.png"

        subprocess.run(["node", "generate_music.js", str(midi_path), emotion, str(intensite)])
        subprocess.run(["timidity", str(midi_path), "-Ow", "-o", str(wav_path)])
        subprocess.run(["ffmpeg", "-y", "-i", str(wav_path), str(mp3_path)])
        subprocess.run(["mscore", str(midi_path), "-o", str(partition_path)])

        prompt = f"an {style} painting illustrating the emotion {emotion}"
        with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
            image = sd_pipe(prompt, num_inference_steps=30).images[0]
        image.save(image_path)

        st.success(f"Émotion détectée : {emotion.capitalize()}")
        st.markdown("---")

        with st.container():
            st.markdown(f"### {titre if titre else 'Résultat généré'}")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Image émotionnelle")
                st.image(str(image_path), use_column_width=True)
                with open(image_path, "rb") as f:
                    st.download_button("Télécharger l'image", f, file_name="illustration.png")

            with col2:
                st.subheader("Musique émotionnelle")
                st.audio(str(mp3_path))
                st.image(str(partition_path), caption="Partition générée", use_column_width=True)
                with open(mp3_path, "rb") as f:
                    st.download_button("Télécharger la musique", f, file_name="musique_emotion.mp3")
                with open(partition_path, "rb") as f:
                    st.download_button("Télécharger la partition", f, file_name="partition.png")

        st.markdown("---")
        st.caption("Merci d'avoir utilisé Illustrer son émotion – image / musique.")