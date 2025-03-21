# IMPORT LIBRARIES
import json
from openai import OpenAI
import numpy as np
import streamlit as st
import os

# CREDENTIALS
API_KEY = os.getenv('OPENAI_API_KEY') 
EMBEDDING_MODEL = "text-embedding-ada-002"
CHATBOT_MODEL = "gpt-4o"
client = OpenAI(api_key=API_KEY)

# SYSTEM MESSAGE
system_message = """
You are an AI assistant that recommends family hotels based only on the provided hotel information.
If the provided hotel information does not cover the user's query, respond with: 'I don’t have enough information to answer this question.'
Only use the provided context and do not invent details.
Be concise, friendly, and helpful in your recommendations.
Anser in GERMAN.
"""

# -----------------UTILITY FUNCTIONS -----------------

def load_articles_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def get_embedding(text, model=EMBEDDING_MODEL):
    response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    embedding = response.data[0].embedding
    return embedding

def find_most_relevant_article(articles, query):
    query_embedding = get_embedding(query)
    best_similarity = -1
    best_article = None
    for article in articles:
        sim = cosine_similarity(query_embedding, article['embedding'])
        if sim > best_similarity:
            best_similarity = sim
            best_article = article
    return best_article, best_similarity

# ----------------- LOAD ALL HOTELS WITH EMBEDDINGS -----------------
HOTELS_FILE = "DATA/family_hotel_with_embeddings.json"
articles = load_articles_from_file(HOTELS_FILE)

# ----------------- STREAMLIT INTERFACE -----------------
st.title("Family Hotel Finder 🏨")

user_query = st.text_input("Wonach suchst du? Beschreibe dein Wunsch-Hotel:")

if st.button("Suche starten"):
    if user_query:
        with st.spinner("Suche nach dem besten passenden Hotel..."):
            best_article, sim = find_most_relevant_article(articles, user_query)

            if best_article:
                context = (
                    f"Hotel Name: {best_article.get('Hotel Name', 'N/A')}\n\n"
                    f"Land: {best_article.get('Land', 'N/A')}\n"
                    f"Region: {best_article.get('Region', 'N/A')}\n\n"
                    f"Typische Familienaktivität: {best_article.get('Typische Familienaktivität', 'N/A')}\n\n"
                    f"Onsite Activities: {best_article.get('Onsite Activities', 'N/A')}\n\n"
                    f"Nearby Attractions: {best_article.get('Nearby Attractions', 'N/A')}\n\n"
                    f"Beschreibung: {best_article.get('Short Description', 'N/A')}\n\n"
                    f"Preisklasse: {best_article.get('Preisklasse', 'N/A')}\n"
                )

                prompt = (
                    f"You are an AI assistant recommending hotels based on the provided hotel data.\n\n"
                    f"Context:\n{context}\n\n"
                    f"User Query: {user_query}\n\n"
                    f"Instructions:\n"
                    f"- Use only the information from the context to answer.\n"
                    f"- If the context is insufficient, answer: 'I don’t have enough information to answer this question.'\n"
                    f"- Be clear, concise, and friendly.\n"
                    f"- Highlight why this hotel could be a good fit.\n\n"
                    f"Recommendation:"
                )

                response = client.chat.completions.create(
                    model=CHATBOT_MODEL,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=300,
                    temperature=0.7
                )
                answer = response.choices[0].message.content

                st.markdown("### Gefundenes Hotel")
                st.markdown(context)
                st.markdown("### Empfehlung")
                st.markdown(answer)
            else:
                st.error("Kein passendes Hotel gefunden.")
    else:
        st.error("Bitte gib eine Beschreibung ein.")
