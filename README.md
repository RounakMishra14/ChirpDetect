# ChirpDetect 🐦

## 📘 Overview
ChirpDetect is an interactive bird sound classification app built with Streamlit and powered by Convolutional Neural Networks (CNNs) to help users identify bird species from their calls. The app allows users to either upload or record audio clips, and then classifies the bird sounds in real time. By leveraging TensorFlow for the machine learning model, Librosa for audio processing, and Plotly for visualization, the app provides a seamless and informative user experience.

The project is built using locally recorded bird call samples, which were limited in number. To compensate for the small dataset size, data augmentation techniques were applied to enhance the training data. Augmentations such as time-stretching, pitch-shifting, and adding background noise helped simulate various real-world scenarios. This enabled the model to better generalize and improve its performance in identifying bird species from audio clips, even in challenging conditions with limited initial recordings.

## 📋 Table of Contents
1. [Overview](#overview)
2. [Demo](#demo)
3. [Data Collection Process](#data-collection-process)
4. [Model Building](#model-building)
5. [App Features](#app-features)
