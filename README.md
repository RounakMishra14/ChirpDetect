# ChirpDetect 🐦
**ChirpDetect** is an interactive bird sound classification app built using Streamlit, leveraging Convolutional Neural Networks (CNNs) to provide users with insights into identifying various bird species based on their calls. The app allows users to visualize results, analyze audio input, and classify bird sounds in real-time.

## 📋 Table of Contents
1. [Overview](#overview)
2. [Demo](#demo)
3. [Data Collection Process](#data-collection-process)
4. [Model Building](#model-building)
5. [App Features](#app-features)

## 📘 Overview
**ChirpDetect** is an interactive bird sound classification app built with Streamlit and powered by Convolutional Neural Networks (CNNs) to help users identify bird species from their calls. The app allows users to upload recordings and then classifies the bird sounds in real time. By leveraging TensorFlow for the machine learning model, Librosa for audio processing, and Plotly for visualization, the app provides a seamless and informative user experience.

The project is built using locally recorded bird call samples from the surrounding area [located here](https://maps.app.goo.gl/45fe6eRNAD3a4nrz6). Due to the limited number of available recordings, **data augmentation** techniques were applied to enhance the dataset. Methods such as time-stretching, pitch-shifting, and adding background noise helped simulate various real-world scenarios. This allowed the model to generalize better and perform effectively, even with the small dataset, by creating a more diverse and robust set of training samples.

## 🌐 Demo
Try the live demo here: [ChirpDetect - Bird Sound Classification App](https://yourdemo.link)

##Here's the **Data Collection** section based on your requirements, including the Wikipedia links and the augmentation details:

---

## 📈 Data Collection Process

To build the bird sound classification model, local bird species were recorded to gather as many samples as possible. The following species, commonly found in the area, were captured:

- [Indian Myna](https://en.wikipedia.org/wiki/Common_myna)
- [Collared Dove](https://en.wikipedia.org/wiki/Eurasian_collared_dove)
- [Kingfisher](https://en.wikipedia.org/wiki/Kingfisher)
- [Owl](https://en.wikipedia.org/wiki/Owl)
- [Sparrow](https://en.wikipedia.org/wiki/House_sparrow)
- [Nightingale](https://en.wikipedia.org/wiki/Common_nightingale)

These recordings were captured from the region [located here](https://maps.app.goo.gl/45fe6eRNAD3a4nrz6). However, due to the limited number of recordings, **data augmentation** techniques were used to increase the size and diversity of the dataset. Augmentation was performed by applying transformations like pitch shifting, time-stretching, and adding background noise, simulating different environments and variations in bird calls.

Here’s the augmentation function used to enhance the dataset:

```python
import librosa
import numpy as np
import os

def augment_audio(input_file, output_folder, pitch_factor=0.5, stretch_factor=0.8, noise_factor=0.005):
    # Load the audio file
    y, sr = librosa.load(input_file)

    # Pitch shifting
    y_pitch = librosa.effects.pitch_shift(y, sr, n_steps=pitch_factor)

    # Time stretching
    y_stretch = librosa.effects.time_stretch(y, stretch_factor)

    # Adding noise
    noise = np.random.randn(len(y)) * noise_factor
    y_noise = y + noise

    # Save augmented files
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    librosa.output.write_wav(os.path.join(output_folder, f"{base_name}_pitch.wav"), y_pitch, sr)
    librosa.output.write_wav(os.path.join(output_folder, f"{base_name}_stretch.wav"), y_stretch, sr)
    librosa.output.write_wav(os.path.join(output_folder, f"{base_name}_noise.wav"), y_noise, sr)

# Example usage:
# augment_audio('input.wav', 'augmented_data/')
```
By augmenting the data, the model is better equipped to handle real-world variations in bird calls, despite having a limited number of original samples.



