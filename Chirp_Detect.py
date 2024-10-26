import streamlit as st
import numpy as np
import librosa
import os
import tensorflow as tf
import plotly.express as px
import pandas as pd

# paths to the dataset
DATA_DIR = 'augmented_data'  
BIRD_CLASSES = ['collared_dove', 'indian_mayna', 'kingfisher', 'nightingale', 'owl', 'sparrow', 'unknown', 'noise']

# Load the trained model
model = tf.keras.models.load_model('bird_classification_model.h5')

def load_wav_16k_mono(filename):
    wav, sample_rate = librosa.load(filename, sr=16000, mono=True)
    return wav

def preprocess(wav):
    # Compute the Short-Time Fourier Transform (STFT)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=-1)  # Add channel dimension
    desired_shape = (150, 257, 1)  # Desired input shape
    spectrogram = tf.image.resize_with_crop_or_pad(spectrogram, desired_shape[0], desired_shape[1])
    return spectrogram

def classify_birds(audio_clips):
    counts = {bird_class: 0 for bird_class in BIRD_CLASSES if bird_class != 'noise'}
    for clip in audio_clips:
        clip_input = preprocess(clip)  # Preprocess the clip
        clip_input = np.expand_dims(clip_input, axis=0)  # Add batch dimension
        prediction = model.predict(clip_input)
        predicted_class_index = np.argmax(prediction)
        predicted_class = BIRD_CLASSES[predicted_class_index]
        if predicted_class != 'noise':
            counts[predicted_class] += 1
    return counts

st.title("ChirpDetect 🐦")

uploaded_file = st.file_uploader("Upload an audio file (MP3 or WAV)", type=['mp3', 'wav'])

if uploaded_file:
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    # Load the uploaded audio file
    audio, sr = librosa.load("temp_audio.wav", sr=None)
    clip_length = 3  # 3 seconds
    num_clips = int(len(audio) // (clip_length * sr))
    audio_clips = [audio[i * clip_length * sr:(i + 1) * clip_length * sr] for i in range(num_clips)]
    
    # Classify the birds
    counts = classify_birds(audio_clips)

    # Filter out categories with zero occurrences and noise
    filtered_counts = {key: value for key, value in counts.items() if value > 0}

    # Convert counts to DataFrame for better table format
    if filtered_counts:
        counts_df = pd.DataFrame(filtered_counts.items(), columns=['Bird Class', 'Count'])
        counts_df['Count'] = counts_df['Count'].astype(int)  # Ensure counts are integers

        # Display the counts in a styled DataFrame
        st.subheader("Bird Class Counts")
        st.dataframe(counts_df.style.highlight_max(axis=0), use_container_width=True)  # Highlight max count in each column

    # Create a pie chart with Plotly
    if filtered_counts:
        fig = px.pie(names=list(filtered_counts.keys()), values=list(filtered_counts.values()),
                      title="Density of Birds in the Clip", hole=0.4)
        fig.update_layout(title_font_size=12)  
        fig.update_traces(textinfo='percent+label')  # Show percentage and label on pie chart
        st.plotly_chart(fig)
