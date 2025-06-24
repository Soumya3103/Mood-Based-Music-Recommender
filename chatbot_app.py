import streamlit as st
import pickle
import pandas as pd
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Load model and vectorizer
model = pickle.load(open('emotion_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Corrected Emotion Map (ensure it aligns with the dataset)
emotion_map = {0: 'sad', 1: 'happy', 2: 'relaxed', 3: 'angry', 4: 'funny', 5: 'confused'}

# Song database
song_dict = {
    'happy': ['Happy - Pharrell Williams', 'Good Vibes - Chris Janson', 'Butta Bomma - Armaan Malik'],
    'sad': ['Someone Like You - Adele', 'Fix You - Coldplay', 'Channa Mereya - Arijit Singh'],
    'angry': ['Breaking the Habit - Linkin Park', 'Bodies - Drowning Pool', 'Macho - Punjabi'],
    'relaxed': ['Weightless - Marconi Union', 'Sunset Lover - Petit Biscuit', 'Pehla Nasha - Udit Narayan'],
    'confused': ['Boulevard of Broken Dreams - Green Day', 'Lose Yourself - Eminem', 'Kabira - Tochi Raina'],
    'funny': ['Never Gonna Give You Up - Rick Astley', 'What Does the Fox Say - Ylvis', 'Aloo Chaat - Kailash Kher']
}

# Friendly chatbot replies
friendly_replies = {
    'happy': "Let's keep the good vibes rolling 🎉",
    'sad': "Feeling low? Here's some songs to lift your spirits 🌈",
    'angry': "Let's calm that fire with some powerful music 🔥",
    'relaxed': "Chill vibes detected 😌 Enjoy this soothing playlist!",
    'confused': "Feeling lost? Music might help you find your way! 🎶",
    'funny': "Let's keep things lighthearted and fun! 😆"
}

# Small talk phrases
small_talk_phrases = ["hello", "hi", "how are you", "what's up", "good morning", "good evening", "hey"]

# Spotify API Setup
client_id = "your_client_id_here"
client_secret = "your_client_secret_here"
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Function to get Spotify link and preview
def get_spotify_link_and_preview(song_name):
    try:
        song_title = song_name.split('-')[0].strip()
        result = sp.search(q=song_title, type='track', limit=1)
        if result['tracks']['items']:
            track = result['tracks']['items'][0]
            return track['external_urls']['spotify'], track['preview_url']
    except Exception as e:
        print(f"Spotify API error: {e}")
    return None, None

# Streamlit App
st.title("🎵 Emotion-Based Song Recommender Chatbot 🤖")
st.write("Hello! I'm your personal music assistant. What can I do for you today?")

user_input = st.text_input("Type something to talk!")

if st.button('Get Song Recommendation'):
    if user_input:
        cleaned_input = user_input.lower().strip()

        # Small talk check
        if any(phrase in cleaned_input for phrase in small_talk_phrases):
            st.info("I'm doing great, thank you! 😊 How can I help you with music today?")
        else:
            # Emotion prediction with debugging
            input_vec = vectorizer.transform([cleaned_input])
            prediction = int(model.predict(input_vec)[0])
            print(f"DEBUG: Raw Prediction Output = {prediction}")  # Check output

            predicted_emotion = emotion_map.get(prediction, "unknown")

            # Friendly message
            friendly_message = friendly_replies.get(predicted_emotion, "Here's some music for you!")
            st.info(friendly_message)

            # Display detected emotion
            st.success(f"**Detected Emotion:** {predicted_emotion.capitalize()}")
            st.write("**Recommended Songs:**")

            # Show songs
            recommended_songs = song_dict.get(predicted_emotion, ['No songs available'])
            for song in recommended_songs:
                song_link, preview_url = get_spotify_link_and_preview(song)
                if song_link:
                    st.write(f"- {song} - [Listen on Spotify]({song_link})")
                    if preview_url:
                        st.audio(preview_url)
                    else:
                        st.write("_Preview not available for this song._")
                else:
                    st.write(f"- {song} - No Spotify link available.")
    else:
        st.warning("Please type something to start the conversation!")
