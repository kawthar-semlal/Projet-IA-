import streamlit as st
import pandas as pd
from openai import OpenAI

# --- CONFIGURATION DE L'IA ---
# Votre clé reste la même
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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", 
)

# On utilise ce modèle stable et gratuit
MODEL_NAME = "google/gemini-2.0-flash-lite-preview-02-05:free"

# --- INTERFACE DU CHATBOT ---
st.set_page_config(page_title="IA Support Client", page_icon="🤖")
st.title("🤖 Chatbot Intelligent - Projet IA 2026")
st.markdown("Structure conseillée pour M. Halim")

# Chargement sécurisé du dataset
@st.cache_data
def load_data():
    try:
        return pd.read_csv('customer_support_tickets_cleaned.csv')
    except:
        # Si le fichier est introuvable, on crée un petit tableau vide pour éviter le crash
        return pd.DataFrame(columns=['Ticket Subject', 'Ticket Description', 'Resolution'])

df = load_data()

# --- HISTORIQUE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- LOGIQUE DU CHATBOT ---
if prompt := st.chat_input("Posez votre question ici..."):
    # 1. Afficher le message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Recherche de contexte (RAG)
    # On cherche si des mots clés existent dans notre dataset
    mots_cles = prompt.lower().split()
    contexte = ""
    if not df.empty:
        # Recherche simple dans les colonnes de texte
        resultats = df[df['Ticket Description'].str.contains('|'.join(mots_cles[:3]), case=False, na=False)].head(2)
        for _, row in resultats.iterrows():
            contexte += f"Historique: {row['Ticket Subject']} -> Solution: {row['Resolution']}\n"

    # 3. Réponse de l'IA
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            instruction_systeme = f"""Tu es un assistant de support client expert et poli.
            Utilise ce contexte issu de nos données si utile : {contexte}
            Si la question n'est pas dans les données, réponds de façon intelligente et aide le client.
            Réponds toujours en français (ou la langue du client)."""

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": instruction_systeme},
                    {"role": "user", "content": prompt}
                ]
            )
            full_response = response.choices[0].message.content
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Désolé, l'IA est très sollicitée. Erreur: {str(e)}")
