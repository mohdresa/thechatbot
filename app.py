import streamlit as st
import pandas as pd
import numpy as np
import os
from gensim.models import Word2Vec
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load the trained Word2Vec model
if os.path.exists('model_retrieval.model'):
    model_w2v = Word2Vec.load('model_retrieval.model')
else:
    st.error("Word2Vec model file not found!")

# Load LSTM-GRU model architecture and weights
try:
    if os.path.exists("model_retrieval_lstm_gru.json") and os.path.exists("model_retrieval_lstm_gru.weights.h5"):
        with open("model_retrieval_lstm_gru.json", "r") as json_file:
            model_json = json_file.read()
        lstm_gru_model = model_from_json(model_json)
        lstm_gru_model.load_weights("model_retrieval_lstm_gru.weights.h5")
        st.write("Model LSTM-GRU loaded successfully.")
    else:
        st.error("LSTM-GRU model files not found!")
except Exception as e:
    st.error(f"Error loading LSTM-GRU model: {e}")

# Load tokenizer
if os.path.exists('tokenizer.pickle'):
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
else:
    st.error("Tokenizer file not found!")

# Load dataset
if os.path.exists('coba.csv'):
    data = pd.read_csv('coba.csv')
    data.rename(columns={'Jawaban': 'jawaban', 'Pertanyaan': 'pertanyaan'}, inplace=True)
else:
    st.error("Data file not found!")

# Define function to create embeddings from Word2Vec model
def get_sentence_vector(sentence, model):
    if isinstance(sentence, str):
        words = [word for word in sentence.split() if word in model.wv]
        if len(words) > 0:
            return np.mean([model.wv[word] for word in words], axis=0)
    return np.zeros(model.vector_size)

# Function to retrieve the most relevant answer based on user question
def retrieve_best_answer(question, model_w2v, data, threshold=0.5):
    # Convert the question to a vector using Word2Vec
    question_vec = get_sentence_vector(question.lower(), model_w2v).reshape(1, -1)

    # Calculate cosine similarity between the question vector and each content vector
    data['Content_Vector'] = data['jawaban'].apply(lambda x: get_sentence_vector(str(x).lower(), model_w2v))
    data['Similarity'] = data['Content_Vector'].apply(lambda x: cosine_similarity(question_vec, x.reshape(1, -1)).flatten()[0])

    # Get the most relevant answer
    best_match = data.sort_values(by='Similarity', ascending=False).iloc[0]

    if best_match['Similarity'] < threshold:
        return "Tidak ada jawaban relevan", "Mohon maaf, saya tidak dapat menemukan jawaban yang sesuai.", 0

    return best_match['pertanyaan'], best_match['jawaban'], best_match['Similarity']

# Function to predict using LSTM-GRU model
def predict_with_lstm_gru(question, tokenizer, lstm_gru_model):
    sequence = tokenizer.texts_to_sequences([question.lower()])
    padded_sequence = pad_sequences(sequence, maxlen=50, padding='post', truncating='post')
    prediction = lstm_gru_model.predict(padded_sequence)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return predicted_class

# Streamlit Chatbot Interface
st.title("Chatbot Retrieval Berbasis Pre-trained Embedding")

# Initialize session state for conversation history
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# User input
user_input = st.text_input("Tanyakan sesuatu:", "")

if st.button("Kirim") and user_input:
    # Log user input
    st.session_state.conversation.append({"role": "user", "message": user_input})

    try:
        # Retrieve the best answer
        best_question, best_answer, similarity = retrieve_best_answer(user_input, model_w2v, data)

        bot_response = (
            f"Berikut adalah jawaban yang paling relevan berdasarkan pertanyaan Anda:\n\n"
            f"**Pertanyaan**: {best_question}\n"
            f"**Jawaban**: {best_answer}\n"
            f"**Tingkat Kemiripan**: {similarity:.2f}\n"
        )

        st.session_state.conversation.append({"role": "bot", "message": bot_response})
    except Exception as e:
        bot_response = "Maaf, saya tidak menemukan jawaban yang relevan dengan pertanyaan tersebut."
        st.session_state.conversation.append({"role": "bot", "message": bot_response})

# Display conversation history
for message in st.session_state.conversation:
    if message["role"] == "user":
        st.write(f"**Anda**: {message['message']}")
    else:
        st.write(f"**Bot**: {message['message']}")
