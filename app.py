import streamlit as st
import pandas as pd
from openai import OpenAI

# --- CONFIGURATION DE L'IA ---
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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", 
)
MODEL_NAME = "meta-llama/llama-3.3-70b-instruct:free"

# --- INTERFACE STREAMLIT ---
st.set_page_config(page_title="IA Customer Support", page_icon="🤖")
st.title("🤖 Assistant Intelligent - Support Client")
st.markdown("---")

# Chargement du dataset
@st.cache_data
def load_data():
    return pd.read_csv('customer_support_tickets_cleaned.csv')

df = load_data()

# Fonction de recherche RAG simple (contexte)
def get_historical_context(query):
    # On cherche les 2 tickets les plus proches par mots-clés
    keywords = query.lower().split()
    mask = df['Ticket Description'].str.contains('|'.join(keywords), case=False, na=False)
    results = df[mask].head(2)
    
    context = ""
    for _, row in results.iterrows():
        context += f"\n- Sujet: {row['Ticket Subject']} | Solution: {row['Resolution']}\n"
    return context if context else "Aucun historique trouvé."

# --- ZONE DE CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage de l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrée utilisateur
if prompt := st.chat_input("Posez votre question (ex: Problème de setup GoPro)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Récupération du contexte RAG
    contexte = get_historical_context(prompt)

    # Appel à l'IA
    with st.chat_message("assistant"):
        instruction_systeme = f"""Tu es un agent de support client expert. 
        Voici le contexte issu de notre base de données : {contexte}
        Réponds de manière professionnelle et concise."""
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": instruction_systeme},
                {"role": "user", "content": prompt}
            ]
        )
        full_response = response.choices[0].message.content
        st.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
