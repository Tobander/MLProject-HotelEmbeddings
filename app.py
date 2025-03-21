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
You are an AI assistant recommending family hotels based only on the provided hotel information.  
Your task is to help the user find the most suitable hotel based on their query.  

Instructions:
- Use only the information from the hotel data (context) provided below.  
- If the hotel matches the request fully or partially, explain clearly why this hotel could be a good fit.  
- If the hotel only partially fits (e.g., not the exact location or not 5 stars), point this out but still present it as a potential option if it has other strengths.  
- If key information (like star rating or wellness facilities) is missing, state that transparently.  
- Always structure your answer in a friendly, clear, and helpful way.  
- Start with a summary sentence (e.g., "I recommend the following hotel for your request:").  
- Then present the details in bullet points:  
   - Location  
   - Star rating (if available)  
   - Wellness and Spa offers  
   - Family activities  
   - Nearby attractions  
   - Price range  
- Finish with a short conclusion explaining whether the hotel is a perfect fit or a close alternative.  
- If no fit is possible, say: "I couldn’t find a perfect match, but here is the closest option I found."
- Answer in GERMAN.
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
                    f"Sterne: {best_article.get('Stern', 'N/A')}\n"
                )

                prompt = (
                    f"Du bist ein KI-Assistent, der basierend auf den unten stehenden Hoteldaten passende Hotels empfiehlt.\n\n"
                    f"Context (Hoteldaten):\n{context}\n\n"
                    f"Benutzeranfrage: {user_query}\n\n"
                    f"Anweisungen:\n"
                    f"- Nutze ausschließlich die Informationen aus dem gegebenen Kontext.\n"
                    f"- Wenn das Hotel vollständig oder teilweise zur Anfrage passt, erkläre freundlich und klar, warum.\n"
                    f"- Wenn es nur teilweise passt (z.B. andere Region, keine Angabe zu Sternen), erwähne das transparent und erkläre, warum es dennoch eine gute Option sein könnte.\n"
                    f"- Fehlen wichtige Informationen, weise darauf hin.\n"
                    f"- Strukturiere deine Antwort in kurzen Abschnitten oder Bulletpoints.\n"
                    f"- Beginne mit einem Satz wie: 'Ich empfehle folgendes Hotel basierend auf deiner Anfrage:'\n"
                    f"- Liste dann Details in Stichpunkten auf (Lage, Sterne, Wellness, Familienaktivitäten, Sehenswürdigkeiten, Preisklasse).\n"
                    f"- Beende die Antwort mit einem kurzen Fazit, ob es eine perfekte Übereinstimmung oder eine gute Alternative ist.\n"
                    f"- Antworte bitte vollständig auf Deutsch.\n\n"
                    f"Empfehlung:"
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
